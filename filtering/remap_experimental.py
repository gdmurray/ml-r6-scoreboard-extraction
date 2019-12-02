from pathlib import Path
import os
import cv2
import time
import csv
import numpy as np
from shutil import copy

headers = ['file_name', 'label', 'x', 'y', 'w', 'h', 'bw_ratio', 'sequence']
training_rows = [headers, ]
training_idx = 1
testing_rows = [headers, ]
testing_idx = 1
for file in os.listdir("sample/"):  # ['frame_467.jpg']:
    file_name = file.split('.')[0]
    print(file_name)
    folder_dir = f"proc_out/{file_name}"
    out_dir = f"text_processing/"
    # Bounding Box Text processing => text2 => cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    # Text Extraction => text4? attempt
    BB_TEXT = "text2"
    EXTRACT_TEXT = "text4"
    for score_idx, score_img in enumerate(sorted(os.listdir(f"{folder_dir}/scorelines/text2"))):
        start = time.time()
        score_name = score_img.split(".")[0]
        print(score_name)
        img_dir = f"{folder_dir}/scorelines/{BB_TEXT}/{score_img}"
        img = cv2.imread(img_dir, 0)

        empty_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        empty_img.fill(255)

        img = 255 - img

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilation = cv2.dilate(img, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        crop_w, crop_y = 0, 0
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

            crop_w, crop_y = w, y
            for x in range(img.shape[1]):
                for y in range(img.shape[0]):
                    if img[y, x] < 2:
                        img[y, x] = 0
                    if img[y, x] > 253:
                        img[y, x] = 255

            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            new_dilation = cv2.dilate(img, rect_kernel, iterations=1)

            contours, hierarchy = cv2.findContours(new_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        IMG_AREA = img.shape[0] * img.shape[1]

        text_img = cv2.imread(f"{folder_dir}/scorelines/{EXTRACT_TEXT}/{score_img}")
        txt_list = ['text1', 'text2', 'text3', 'text4']
        text_img_map = {txt: cv2.imread(f"{folder_dir}/scorelines/{txt}/{score_img}", 0) for txt in txt_list}

        if crop_y != 0 or crop_w != 0:
            for tx in text_img_map:
                tmp_img = text_img_map[tx]
                text_img_map[tx] = tmp_img[0 + crop_y:tmp_img.shape[0], 0 + crop_w:tmp_img.shape[1]]
        i = 1
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w > 8:
                img_file = text_img_map['text2']
                file_h, file_w = text_img_map['text2'].shape
                # print(file_w, file_h)
                # print(x, y, w, h, i)
                original_file_name = f"text_{file_name}_{score_idx}_{i}.jpg"
                original_file_location = f"text_processing/{file_name}/{score_name}/{original_file_name}"
                # print(original_file_location + original_file_name)
                PATHS = ["filtering/training", "filtering/testing"]
                for PATH in PATHS:
                    found = [filename for filename in Path(PATH).rglob(original_file_name)]
                    if len(found) > 0:
                        if PATH == "filtering/testing":
                            print("Found in testing")
                        file_path = found[0]
                        label = str(file_path).split("/")[-2]
                        filename = str(file_path).split("/")[-1]

                        bound_img = img_file[y:y + h, x:x + w]
                        pixels = bound_img.shape[0] * bound_img.shape[1]
                        non_zero = cv2.countNonZero(bound_img)
                        ratio = 1 - (non_zero / pixels)
                        x_ratio = (x / file_w)
                        w_ratio = (w / file_w)
                        y_ratio = (y / file_h)
                        h_ratio = (y / file_h)
                        if PATH == "filtering/training":
                            training_rows.append([])
                            training_rows[training_idx] = [filename, label, x_ratio, y_ratio, w_ratio, h_ratio, ratio, i]
                            training_idx += 1
                            if not os.path.exists(f"filtering/experimental/training/{label}"):
                                os.makedirs(f"filtering/experimental/training/{label}")

                            copy(original_file_location, f"filtering/experimental/training/{label}")
                        elif PATH == "filtering/testing":
                            testing_rows.append([])
                            testing_rows[testing_idx] = [filename, label, x_ratio, y_ratio, w_ratio, h_ratio, ratio, i]
                            testing_idx += 1

                            if not os.path.exists(f"filtering/experimental/testing/{label}"):
                                os.makedirs(f"filtering/experimental/testing/{label}")

                            copy(original_file_location, f"filtering/experimental/testing/{label}")
                i += 1

output_file = open("training_remapped.csv", 'w')
output_test = open("testing_remapped.csv", 'w')
writer = csv.writer(output_file)
t_writer = csv.writer(output_test)
writer.writerows(training_rows)
t_writer.writerows(testing_rows)
