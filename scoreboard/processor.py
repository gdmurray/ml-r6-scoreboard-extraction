import cv2
import base64
import pickle
import os
import csv
import requests
import concurrent.futures
import json
import math
import copy
from pytesseract import image_to_data
import time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from filtering.functions import resize_image


class Processor:
    MIN_SCORELINE_HEIGHT = 10
    TEXT_LIST = ['text1', 'text2', 'text3', 'text4']
    CSV_HEADERS = ['num', 'text1', 'text2', 'text3', 'text4', 'classification', "exp_classification"]

    def __init__(self, image, image_name, cache=True):
        self.root_image = image
        self.image_name = image_name
        self.cache_folder = f"scoreboard/cache/{image_name.split('.')[0]}"
        self.out_folder = f"scoreboard/out/{image_name.split('.')[0]}"
        self.cache = cache
        self.init_folders()

    def init_folders(self):
        if not os.path.exists(self.cache_folder):
            os.mkdir(self.cache_folder)

        if not os.path.exists(self.out_folder):
            os.mkdir(self.out_folder)

    def extract_scoreboard_box(self, image=None):
        img = self.root_image
        cache_dir = f"{self.cache_folder}/extract"

        print("Thresholding Image")
        th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        th1 = 255 - th1

        cv2.imwrite(f"{cache_dir}/adaptive.jpg", th1)

        kernel_length = np.array(img).shape[1] // 80

        # Get Horizontal And Vertical Lines from Image
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        img_temp1 = cv2.erode(th1, v_kernel, iterations=3)
        vertical_lines_img = cv2.dilate(img_temp1, v_kernel, iterations=15)
        if self.cache: cv2.imwrite(f"{cache_dir}/vertical_lines.jpg", vertical_lines_img)

        img_temp2 = cv2.erode(th1, h_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, h_kernel, iterations=15)
        if self.cache: cv2.imwrite(f"{cache_dir}/horizontal_lines.jpg", horizontal_lines_img)

        # Combine Images
        alpha = 0.5
        beta = 1.0 - alpha
        img_final_bin = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=1)
        if self.cache: cv2.imwrite(f"{cache_dir}/img_pre_thresh.jpg", img_final_bin)

        (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if self.cache: cv2.imwrite(f"{cache_dir}/img_final_bin.jpg", img_final_bin)

        # Fetch Contours from Image
        print("Fetching Contours Image")
        contours, _ = cv2.findContours(img_final_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        first_contour_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        first_contour_img.fill(255)

        i = 0
        for c in contours:
            i += 1
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

            if x > 100 and ((x + w) < img.shape[1]) and y > 100:
                cv2.rectangle(first_contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if self.cache: cv2.imwrite(f"{cache_dir}/approx_contours.jpg", first_contour_img)

        # Pass two of getting vertical lines from image
        img_temp1_pass2 = cv2.erode(first_contour_img, v_kernel, iterations=3)
        vertical_lines_img_pass2 = cv2.dilate(img_temp1_pass2, v_kernel, iterations=3)

        img_temp2_pass2 = cv2.erode(first_contour_img, h_kernel, iterations=3)
        horizontal_lines_img_pass2 = cv2.dilate(img_temp2_pass2, h_kernel, iterations=3)

        alpha = 0.5
        beta = 1.0 - alpha
        img_final_bin_pass2 = cv2.addWeighted(vertical_lines_img_pass2, alpha, horizontal_lines_img_pass2, beta, 0.0)
        img_final_bin_pass2 = cv2.erode(~img_final_bin_pass2, kernel, iterations=1)
        img_final_bin_pass2 = cv2.dilate(img_final_bin_pass2, kernel, iterations=2)

        (thresh, img_final_bin_pass2) = cv2.threshold(img_final_bin_pass2, 128, 255,
                                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if self.cache:
            cv2.imwrite(f"{cache_dir}/img_final_bin_pass2.jpg", img_final_bin_pass2)

        # Get Contours from Second Pass on Image
        contours_pass2, _ = cv2.findContours(img_final_bin_pass2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract Edges
        edges = cv2.Canny(img_final_bin_pass2, 100, 200, apertureSize=3)

        # Extract Corners from Edges
        corners = cv2.goodFeaturesToTrack(edges, 30, 0.01, 100)

        corner_points = [(c.ravel()[0], c.ravel()[1]) for c in corners]
        corners = np.int0(corners)

        validation_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        validation_img.fill(255)

        for corn in corners:
            x, y = corn.ravel()
            cv2.circle(validation_img, (x, y), 5, (0, 255, 0), -1)

        if self.cache:
            cv2.imwrite(f"{cache_dir}/corners.jpg", validation_img)
        if self.cache:
            cv2.imwrite(f"{cache_dir}/edges.jpg", edges)

        # Extract Hough Lines from Image (detectable lines)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)

        horizontal_lines = []
        vertical_lines = []

        # Iterate Through Detected Houghlines
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + img.shape[1] * (-b))
                y1 = int(y0 + img.shape[0] * (a))
                x2 = int(x0 - img.shape[1] * (-b))
                y2 = int(y0 - img.shape[0] * (a))
                abs_x = abs(abs(x1) - abs(x2))
                abs_y = abs(abs(y1) - abs(y2))

                # Only Accept Straight Slanted Lines
                if abs_x <= 1 and abs_y <= 1:
                    if a < 0:
                        h_check = [(abs(c[1] - y2) < 3) for c in corner_points]
                        # Check if point intersects with corners
                        if any(h_check):
                            cv2.line(validation_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            horizontal_lines.append([(x1, y1), (x2, y2)])
                    else:
                        v_check = [(abs(c[0] - x2) < 5) for c in corner_points]
                        # Check if point intersects with corners
                        if any(v_check):
                            cv2.line(validation_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            vertical_lines.append([(x1, y1), (x2, y2)])

        if self.cache:
            cv2.imwrite(f"{cache_dir}/houghlines.jpg", validation_img)

        # Init Empty image for bounding boxes
        bounding_box = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        bounding_box.fill(255)

        # If more than 1 contour in pass2, take the one with the most area
        if len(contours_pass2) > 1:
            contour_p2 = max(contours_pass2, key=lambda x: cv2.boundingRect(x)[2] * cv2.boundingRect(x)[3])
        else:
            contour_p2 = contours_pass2[0]

        # Approximate perimeter of area
        peri = cv2.arcLength(contour_p2, True)
        approx = cv2.approxPolyDP(contour_p2, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)

        cv2.rectangle(bounding_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if self.cache:
            cv2.imwrite(f"{cache_dir}/bounding_box.jpg", bounding_box)

        # Sort by corner lines to restrict the bounds of the box
        sorted_horizontal_lines = sorted(horizontal_lines, key=lambda x: x[1][1])
        sorted_vertical_lines = sorted(vertical_lines, key=lambda x: x[1][0])

        uppermost_horizontal_line = sorted_horizontal_lines[0]
        lowermost_horizontal_line = sorted_horizontal_lines[-1]

        leftmost_vertical_line = sorted_vertical_lines[0]
        rightmost_vertical_line = sorted_vertical_lines[-1]

        # Y = BOUNDING BOX UPPER Y AXIS
        print(f"Pre Bounds: x:{x} y:{y} w:{w} h:{h}")
        if y < uppermost_horizontal_line[1][1]:
            y = uppermost_horizontal_line[1][1]

        if (y + h) > lowermost_horizontal_line[1][1]:
            h = lowermost_horizontal_line[1][1] - y

        if x < leftmost_vertical_line[0][0]:
            x = leftmost_vertical_line[0][0]

        if (x + w) > rightmost_vertical_line[0][0]:
            w = rightmost_vertical_line[0][0] - x

        print(f"Final Bounds: x:{x} y:{y} w:{w} h:{h}")
        print()
        final_bounds_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        final_bounds_img.fill(255)

        cv2.rectangle(final_bounds_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if self.cache:
            cv2.imwrite(f"{cache_dir}/final_bounds.jpg", final_bounds_img)

        cropped = img[y:y + h, x:x + w]
        if self.cache:
            cv2.imwrite(f"{cache_dir}/extracted_scoreboard.jpg", cropped)

        if not os.path.exists(f"{self.out_folder}/scoreboard"):
            os.mkdir(f"{self.out_folder}/scoreboard")
        cv2.imwrite(f"{self.out_folder}/scoreboard/extracted_scoreboard.jpg", cropped)

        return cropped

    def extract_scorelines(self, img):
        cache_dir = f"{self.cache_folder}/process"

        cv2.imwrite(f"{cache_dir}/original.jpg", img)

        # Extract Scoreboard with 4 different methods
        _, text_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, text_img_2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
        text_img_3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4)
        _, text_img_4 = cv2.threshold(img, 80, 255, cv2.THRESH_TOZERO)

        # Invert text1
        text_img = 255 - text_img

        # invert text2
        text_img_2 = 255 - text_img_2

        text_map = {
            "text1": text_img,
            "text2": text_img_2,
            "text3": text_img_3,
            "text4": text_img_4
        }

        if self.cache:
            cv2.imwrite(f"{cache_dir}/text1.jpg", text_img)
            cv2.imwrite(f"{cache_dir}/text2.jpg", text_img_2)
            cv2.imwrite(f"{cache_dir}/text3.jpg", text_img_3)
            cv2.imwrite(f"{cache_dir}/text4.jpg", text_img_4)

        # Threshold original image for easier line detection
        _, line_img = cv2.threshold(img, 50, 255, cv2.THRESH_TRUNC)

        if self.cache: cv2.imwrite(f"{cache_dir}/pre_line.jpg", line_img)

        # Thresh line img again
        _, line_img = cv2.threshold(line_img, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if self.cache: cv2.imwrite(f"{cache_dir}/lines.jpg", line_img)

        edges = cv2.Canny(line_img, 50, 75, apertureSize=3)

        if self.cache: cv2.imwrite(f"{cache_dir}/edges.jpg", edges)

        dilation_kernel = np.ones((2, 2), np.uint8)
        dilated_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                         dilation_kernel)  # cv2.dilate(edges, d_kernel, iterations=2)

        if self.cache:  cv2.imwrite(f"{cache_dir}/dilated.jpg", dilated_edges)

        houghlines_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        houghlines_img.fill(255)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        ct = 0

        # Iterate over found Houghlines
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + img.shape[1] * (-b))
                y1 = int(y0 + img.shape[0] * (a))
                x2 = int(x0 - img.shape[1] * (-b))
                y2 = int(y0 - img.shape[0] * (a))
                abs_x = abs(abs(x1) - abs(x2))
                abs_y = abs(abs(y1) - abs(y2))
                # Only take Horizontal and Vertical Lines
                if abs_x <= 1 and abs_y <= 1:
                    if b == 1.0:
                        # print(f"a: {a} b: {b} ({x1},{y1}) -> ({x2},{y2}) absx: {abs_x} absy: {abs_y}")
                        ct += 1
                        cv2.line(houghlines_img, (x1, y1), (x2, y1), (0, 0, 255), 2)
                    elif b == 0.0:
                        ct += 1
                        cv2.line(houghlines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if self.cache: cv2.imwrite(f"{cache_dir}/houghlines.jpg", houghlines_img)

        w_counter = Counter()
        h_counter = Counter()

        w_e_counter = Counter()
        h_e_counter = Counter()

        pre_contour_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        pre_contour_img.fill(255)

        # Find Contours in Houghlines Image
        contours, hierarchy = cv2.findContours(houghlines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            w_counter[w] += 1
            h_counter[h] += 1
            cv2.rectangle(pre_contour_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        if self.cache:  cv2.imwrite(f"{cache_dir}/pre_contour.jpg", pre_contour_img)

        ranges = []
        height_list = sorted(h_counter)

        i = 0
        prev = height_list[0]

        # find common height and use that as player line baseline
        while i < len(height_list) - 1:
            if height_list[i] > self.MIN_SCORELINE_HEIGHT:
                if abs(height_list[i] - height_list[i + 1]) <= 2:
                    j = i
                    rng = [height_list[j]]
                    # While the difference between continuous height boxes are <= 2
                    while (j < len(height_list) - 1) and abs(height_list[j] - height_list[j + 1]) <= 2:
                        rng.append(height_list[j + 1])
                        j += 1
                    ranges.append(rng)
                    i = j
            i += 1

        # Extract range with most continuous features
        accepted_range = max(ranges, key=lambda p: len(p))

        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h in accepted_range:
                filtered_contours.append([x, y, w, h])

        # Sort contours by X location
        filtered_contours = sorted(filtered_contours, key=lambda cn: cn[0])

        # Final List of Data
        data_lines = defaultdict(list)

        # Add Filter
        for cont in filtered_contours:
            data_lines[cont[1]].append(cont)

        # If algorithm missed scorelines, approximate location
        while len(data_lines) < 10:
            y_keys = sorted(data_lines.keys())
            diffs = []
            diff_map = {}
            # Calculate differences between Scorelines
            for i in range(len(data_lines) - 1):
                bottom_this = data_lines[y_keys[i]][0][1] + data_lines[y_keys[i]][0][3]
                top_next = data_lines[y_keys[i + 1]][0][1]
                diff = abs(bottom_this - top_next)
                diffs.append(diff)
                diff_map[diff] = y_keys[i]

            max_diff = max(diffs)
            i = 0
            y_ranges = []
            diffs = sorted(diffs)

            # Get Continuous Y-Ranges of Boxes
            while i < len(diffs) - 1:
                if abs(diffs[i] - diffs[i + 1]) <= 3:
                    j = i
                    rng = [diffs[j]]
                    while (j < len(diffs) - 1) and abs(diffs[j] - diffs[j + 1]) <= 2:
                        rng.append(diffs[j + 1])
                        j += 1
                    y_ranges.append(rng)
                    i = j
                i += 1
            # Accepted offset range is tuple with max features
            accepted_offset = max(y_ranges, key=lambda p: len(p))

            avg_diff = math.floor(sum(set(accepted_offset)) / len(set(accepted_offset)))
            avg_height = math.floor(sum(set(accepted_range)) / len(set(accepted_range)))

            new_y = data_lines[diff_map[max_diff]][0][1] + data_lines[diff_map[max_diff]][0][3] + avg_diff

            # Append new approximated scoreline to datalines
            for rect in data_lines[diff_map[max_diff]]:
                data_lines[new_y].append([rect[0], new_y, rect[2], avg_height])

        # Create Finalized Scoreline Rects
        final_rects = []

        contour_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        contour_img.fill(255)

        for y, contours in data_lines.items():
            for cont in contours:
                x, y, w, h = cont
                cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            final_rects.append([
                contours[0][0],
                contours[0][1],
                sum([c[2] for c in contours]),
                contours[0][3]
            ])

        final_rect_image = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        final_rect_image.fill(255)

        # Sort Scorelines by Y location to get accurate positioning
        for rect_idx, final in enumerate(sorted(final_rects, key=lambda x: x[1])):
            score_folder = self.out_folder + f"/score_{rect_idx}"
            if not os.path.exists(score_folder):
                os.makedirs(score_folder)

            x, y, w, h = final
            cv2.rectangle(final_rect_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            pre_crop = Processor.clean_edge_cases(text_map["text2"])

            for text, text_image in text_map.items():
                text_directory = f"{score_folder}/text"
                if not os.path.exists(text_directory):
                    os.makedirs(text_directory)
                score_line = text_image[y + pre_crop:y + h, x:x + w]
                cv2.imwrite(f"{text_directory}/{text}.jpg", score_line)

        if self.cache: cv2.imwrite(f"{cache_dir}/final_rect.jpg", final_rect_image)

    def extract_scores(self):
        prediction_dict = {
            "file": self.image_name,
            "scores": []
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            scores = sorted([score for score in os.listdir(f"{self.out_folder}") if score != "scoreboard"])

            args = (score_idx for score_idx in range(0, len(scores)))
            for result in executor.map(self.process_scoreline, args):
                print(result)
                # prediction_dict["scores"].append(result)
        # json.dump(prediction_dict, open(f"{self.out_folder}/scoreboard/text.json", "w"))

    def process_scoreline(self, score_idx):
        os.environ['TESSDATA_PREFIX'] = "/usr/share/tesseract-ocr"
        if not os.path.exists(self.cache_folder + "/textraction"):
            os.mkdir(self.cache_folder + "/textraction")
        cache_dir = self.cache_folder + "/textraction"
        text_img_map = {text: cv2.imread(f"{self.out_folder}/score_{score_idx}/text/{text}.jpg", 0) for text in
                        self.TEXT_LIST}

        img = text_img_map["text2"]

        start = time.time()
        score_name = f"score_{score_idx}"

        # Invert Image
        img = 255 - img

        if self.cache: cv2.imwrite(f"{cache_dir}/original_{score_name}.jpg", img)

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilation = cv2.dilate(img, rect_kernel, iterations=1)

        if self.cache:
            cv2.imwrite(f"{cache_dir}/dilation_{score_name}.jpg", dilation)

        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        crop_w, crop_y = 0, 0

        # Some sort of invalid artifact if only 1 contour, perform operations
        if len(contours) == 1:
            # Crop operations...
            w_midpoint = img.shape[1] // 2
            h_midpoint = img.shape[0] // 2

            w, y = 0, 0
            while img[h_midpoint, w] > 250:
                w += 1
            while img[y, w_midpoint] > 250:
                y += 1
            img = img[0 + y:img.shape[0], 0 + w:img.shape[1]]

            crop_w += w
            crop_y += y
            # Custom threshold function
            for x in range(img.shape[1]):
                for y in range(img.shape[0]):
                    if img[y, x] < 2:
                        img[y, x] = 0
                    if img[y, x] > 253:
                        img[y, x] = 255

            if self.cache: cv2.imwrite(f"{cache_dir}/original_{score_name}.jpg", img)

            new_dilation = cv2.dilate(img, rect_kernel, iterations=1)

            if self.cache: cv2.imwrite(f"{cache_dir}/dilation_{score_name}.jpg", dilation)

            contours, hierarchy = cv2.findContours(new_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print("New Contours: ", len(contours))

        contour_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        contour_img.fill(255)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        if self.cache: cv2.imwrite(f"{cache_dir}/contours_{score_name}.jpg", contour_img)

        if crop_y != 0 or crop_w != 0:
            for t, t_img in text_img_map.items():
                tmp_img = text_img_map[t]
                text_img_map[t] = tmp_img[0 + crop_y:tmp_img.shape[0], 0 + crop_w:tmp_img.shape[1]]

        i = 1
        # Sort Contours by X Location across image
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

        rows = [self.CSV_HEADERS, ]
        conf_table = [self.CSV_HEADERS[:-2], ]

        text_file = open(f"{self.out_folder}/score_{score_idx}/text.csv", 'w')
        confidence_file = open(f"{self.out_folder}/score_{score_idx}/conf.csv", 'w')

        text_writer = csv.writer(text_file)
        conf_writer = csv.writer(confidence_file)

        print(f"Found Contours: {len(contours)}")
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w > 8:
                rows.append([i, ])
                conf_table.append([i, ])
                text2_img_og = text_img_map['text2']
                text2_img = text2_img_og[y:y + h, x:x + w]
                resized = np.array(resize_image(text2_img), dtype=np.float64)
                exp_array = Processor.get_exp_params(x, y, w, h, text2_img_og, text2_img, i).astype(np.float64)

                image_param = base64.urlsafe_b64encode(resized.tobytes()).decode('utf-8')
                exp_param = base64.urlsafe_b64encode(exp_array.tobytes()).decode('utf-8')

                url = f"http://localhost/predict?image={image_param}&exp={exp_param}"
                response = requests.get(url)
                print(response.text)
                response_data = response.json()
                # print(response.content, response.text)

                # prediction = "name"
                # exp_prediction = "name"
                prediction = response_data["prediction"]
                exp_prediction = response_data["exp_prediction"]

                box_crop = 0
                if exp_prediction == "name" and i < 3:
                    name_img = copy.deepcopy(text2_img)
                    thresh = cv2.threshold(name_img, 160, 255, cv2.THRESH_BINARY_INV)[1]
                    name_dilation = cv2.dilate(thresh, rect_kernel, iterations=2)
                    b_contours, b_hierarchy = cv2.findContours(name_dilation, cv2.RETR_EXTERNAL,
                                                               cv2.CHAIN_APPROX_NONE)
                    c_img = np.zeros([text2_img.shape[0], text2_img.shape[1]], dtype=np.uint8)
                    c_img.fill(255)
                    name_contours = [cv2.boundingRect(b_cont) for b_cont in b_contours]
                    box_contour = min(name_contours, key=lambda box: abs(box[2] - box[3]))
                    box_crop = box_contour[0] + box_contour[2]
                    cv2.imwrite(f"{self.out_folder}/score_{score_idx}/name_contours.jpg", c_img)

                print("Predicting Text Data")
                for t in self.TEXT_LIST:
                    if (y + h) >= img.shape[0]:
                        bound_img = np.array(t_img[y:img.shape[0], x + box_crop:x + w + box_crop])
                    else:
                        bound_img = np.array(t_img[y:y + h, x + box_crop:x + w])
                    if i >= 3 or prediction in ['number']:
                        data = image_to_data(bound_img, lang="eng_best", config='--psm 10 --oem 3',
                                             output_type='dict')
                    else:
                        data = image_to_data(bound_img, lang="eng_best", config='--psm 12', output_type='dict')
                    dataframe = pd.DataFrame(data)

                    dataframe['conf'] = dataframe['conf'].astype(int)
                    lines = dataframe.groupby('block_num')['text'].apply(list)
                    line = max(lines, key=lambda l: len("".join([s for s in l if s != ''])))
                    data_string = ''.join([l for l in line])
                    block_lines = {"".join([s for s in value if s != '']): key for key, value in lines.items()}
                    block_num = block_lines[data_string]
                    conf_df = dataframe.groupby(['block_num', 'conf'])['text'].apply(list)
                    # print(conf_df)
                    conf_keys = [k for k in conf_df[block_num].keys() if k != -1]
                    conf_string = "".join(Processor.flatten([conf_df[block_num][k] for k in conf_keys]))
                    if len(conf_keys) > 0:
                        conf_val = max(conf_keys)
                    else:
                        conf_val = -1
                    # if conf_val != -1:
                    # print(conf_string, conf_val)
                    string = "".join([s for s in conf_string if str.isalnum(s) or s in ['_', '-', '.']])
                    rows[i].append(string)
                    conf_table[i].append(conf_val)

                rows[i].append(prediction)
                rows[i].append(exp_prediction)
                # cv2.imwrite(f"text_processing/{file_name}/{score_name}/text_{file_name}_{score_idx}_{i}.jpg",
                #            text2_img)
                i += 1

            text_writer.writerows(rows)
            conf_writer.writerows(conf_table)
            finish = time.time()

            score_dict = {
                "order": score_idx,
                "deleted": []
            }
            d_keys = ["name", "score", "kills", "assists", "deaths"]
            for key in d_keys:
                score_dict[key] = {"value": None, "alternatives": []}

            pop_idx = {}
            for r in range(1, len(rows)):
                if rows[r][5] == "side" and ((len(set(rows[r][1:5])) == 1 and rows[r][1] == '') or r == 1):
                    # print(f"Found Side Symbol... skipping row {r}")
                    pop_idx[r] = "Side Symbol"
                elif rows[r][5] == "other" and rows[r][6] == "name":
                    # print(f"Found Other... skipping row {r}")
                    pop_idx[r] = "Other (Rounds?)"
                elif rows[r][-1] == "icon" and rows[r][-2] == rows[r][-1]:
                    # print(f"Icon Detected... skipping row: {r}")
                    pop_idx[r] = "Icon"
                elif (len("".join(rows[r][1:5])) <= 8 and (
                        rows[r][5] == "number" and rows[r][6] in ['icon', 'number']) and r < 4) or sum(
                    conf_table[r][1:5]) == -4:
                    pop_idx[r] = "Pre-Name Icon Removal"

            delete_rows = sorted(pop_idx.keys())
            delete_rows.reverse()

            for idx in delete_rows:
                del rows[idx]
                del conf_table[idx]

            def convert_val(v):
                if v in ["o", "O"]:
                    return 0
                return v

            for final_idx, key in enumerate(d_keys):
                r_idx = final_idx + 1
                if r_idx < len(rows):
                    print(rows[r_idx])
                    print(conf_table[r_idx])
                    alternatives = sorted([(val, conf_table[r_idx][i + 1]) for i, val in enumerate(rows[r_idx][1:5]) if
                                           str(val) and i != 3], key=lambda x: x[1], reverse=True)
                    if final_idx > 1:
                        digit_alternatives = sorted(
                            [(int(val), conf_table[r_idx][i + 1]) for i, val in enumerate(rows[r_idx][1:5]) if
                             str.isdigit(str(val)) and i != 3], key=lambda x: x[1], reverse=True)
                        if str.isdigit(rows[r_idx][2]):
                            score_dict[key]["value"] = (int(rows[r_idx][2]), conf_table[r_idx][2])
                            score_dict[key]["alternatives"] = digit_alternatives[:2]
                        else:
                            convert_attempt = sorted(
                                [(convert_val(v[0]), v[1]) for v in alternatives if type(convert_val(v)) == 0],
                                key=lambda x: x[1], reverse=True)

                            if len(convert_attempt) > 0:
                                score_dict[key]["value"] = convert_attempt[0][0]

                    if len(alternatives) > 0:
                        if not score_dict[key]["value"]:
                            score_dict[key]["value"] = alternatives[:1]
                        score_dict[key]["alternatives"] = alternatives[1:3]
                    else:
                        score_dict[key]["value"] = ("-", -1)

                # Remove duplicates
                # score_dict[key]["alternatives"] = [score_dict[key]["alternatives"][p] for p in
                #                                   score_dict[key]["alternatives"] if
                #                                   str(score_dict[key]["alternatives"][p][0]) != str(
                #                                       score_dict[key]["value"][0])]
            score_dict["deleted"] = delete_rows
            print(f"{score_name} took {finish - start:.2f}s")
            return json.dumps(score_dict)

    @staticmethod
    def clean_edge_cases(image):
        image = 255 - image
        rows = []
        crop_y = 0
        for y in range(0, 3):
            rows.append([])
            for x in range(image.shape[1]):
                rows[y].append(image[y, x])
        if any((len([r for r in row if r < 127]) / image.shape[1]) > 0.25 for row in rows):
            crop_y = 3
        return crop_y

    @staticmethod
    def get_exp_params(x, y, w, h, file_, sub_img, seq):
        file_h, file_w = file_.shape

        pixels = sub_img.shape[0] * sub_img.shape[1]
        non_zero = cv2.countNonZero(sub_img)
        ratio = 1 - (non_zero / pixels)

        x_ratio = (x / file_w)
        w_ratio = (w / file_w)
        y_ratio = (y / file_h)
        h_ratio = (y / file_h)
        return np.array([x_ratio, y_ratio, w_ratio, h_ratio, ratio, seq])

    @staticmethod
    def flatten(S):
        if S == []:
            return S
        if isinstance(S[0], list):
            return Processor.flatten(S[0]) + Processor.flatten(S[1:])
        return S[:1] + Processor.flatten(S[1:])
