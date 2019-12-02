import cv2
import numpy as np
import os
from collections import defaultdict, Counter
import math


def clean_edge_cases(image, path):
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


MIN_HEIGHT = 10
valid = []
for file in ['frame_1545.jpg']:  # os.listdir("sample/"):
    print(file)
    img = cv2.imread(f"out/{file.split('.')[0]}/final_scoreboard.jpg", 0)
    folder_dir = f"proc_out/{file.split('.')[0]}"
    delete = ["close_lines.jpg", "open_text.jpg", "vertical.jpg"]

    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    if not os.path.exists(f"{folder_dir}/scorelines"):
        os.makedirs(f"{folder_dir}/scorelines")

    if not os.path.exists(f"{folder_dir}/text"):
        os.makedirs(f"{folder_dir}/text")

    for d in delete:
        if os.path.exists(f"{folder_dir}/{delete}"):
            os.remove(f"{folder_dir}/{delete}")

    cv2.imwrite(f"{folder_dir}/original.jpg", img)

    TEXT_THRESH = cv2.THRESH_BINARY
    LINE_THRESH = cv2.THRESH_TRUNC

    (_, text_img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    (_, text_img_2) = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    (_, text_img_4) = cv2.threshold(img, 80, 255, cv2.THRESH_TOZERO)
    text_img_3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4)

    # text_img =
    text_img = 255 - text_img
    text_img_2 = 255 - text_img_2
    text_map = {
        "text1": text_img,
        "text2": text_img_2,
        "text3": text_img_3,
        "text4": text_img_4
    }
    cv2.imwrite(f"{folder_dir}/text/text1.jpg", text_img)
    cv2.imwrite(f"{folder_dir}/text/text2.jpg", text_img_2)
    cv2.imwrite(f"{folder_dir}/text/text3.jpg", text_img_3)
    cv2.imwrite(f"{folder_dir}/text/text4.jpg", text_img_4)
    # t_kernel = np.ones((1, 1), np.uint8)
    # text_img = cv2.erode(text_img, t_kernel, iterations=5)

    (l_thresh, line_img) = cv2.threshold(img, 50, 255, cv2.THRESH_TRUNC)
    cv2.imwrite(f"{folder_dir}/pre_line.jpg", line_img)
    (b_thresh, line_img) = cv2.threshold(line_img, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # line_img = cv2.adaptiveThreshold(line_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)
    # cv2.imwrite(f"processing/test/adapt.jpg", adapt_img)

    # kernel_length = np.array(img).shape[1] // 80
    # h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # h_temp = cv2.erode(line_img, h_kernel, iterations=3)
    # horizontal_img = cv2.dilate(h_temp, h_kernel, iterations=15)
    # cv2.imwrite(f"{folder_dir}/horizontal.jpg", horizontal_img)
    # line_img = 255 - line_img

    cv2.imwrite(f"{folder_dir}/lines.jpg", line_img)

    # kernel = np.ones((2, 2), np.uint8)
    # close_lines = cv2.morphologyEx(line_img, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(f"{folder_dir}/close_lines.jpg", line_img)

    edges = cv2.Canny(line_img, 50, 75, apertureSize=3)
    cv2.imwrite(f"{folder_dir}/edges.jpg", edges)

    d_kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, d_kernel)  # cv2.dilate(edges, d_kernel, iterations=2)
    cv2.imwrite(f"{folder_dir}/dilated.jpg", dilated_edges)

    empty_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    empty_img.fill(255)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    ct = 0
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
            if abs_x <= 1 and abs_y <= 1:
                if b == 1.0:
                    # print(f"a: {a} b: {b} ({x1},{y1}) -> ({x2},{y2}) absx: {abs_x} absy: {abs_y}")
                    ct += 1
                    cv2.line(empty_img, (x1, y1), (x2, y1), (0, 0, 255), 2)
                elif b == 0.0:
                    ct += 1
                    cv2.line(empty_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    print(f"Lines: {len(lines)}->{ct}")
    cv2.imwrite(f"{folder_dir}/houghlines.jpg", empty_img)

    w_counter = Counter()
    h_counter = Counter()

    w_e_counter = Counter()
    h_e_counter = Counter()

    pre_contour_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    pre_contour_img.fill(255)
    contour_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    contour_img.fill(255)

    contours, hierarchy = cv2.findContours(empty_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        w_counter[w] += 1
        h_counter[h] += 1
        cv2.rectangle(pre_contour_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imwrite(f"{folder_dir}/pre_contour.jpg", pre_contour_img)
    ranges = []
    dropped = []
    h_list = sorted(h_counter)
    i = 0
    prev = h_list[0]

    # Find common height and use as player line
    while i < len(h_list) - 1:
        if h_list[i] < MIN_HEIGHT:
            dropped.append(h_list[i])
        else:
            if abs(h_list[i] - h_list[i + 1]) <= 2:
                j = i
                rng = [h_list[j]]
                while (j < len(h_list) - 1) and abs(h_list[j] - h_list[j + 1]) <= 2:
                    rng.append(h_list[j + 1])
                    j += 1
                ranges.append(rng)
                i = j
        i += 1

    accepted_range = max(ranges, key=lambda p: len(p))

    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h in accepted_range:
            filtered_contours.append([x, y, w, h])
            # cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    filtered_contours = sorted(filtered_contours, key=lambda cn: cn[0])
    data_lines = defaultdict(list)

    for cont in filtered_contours:
        data_lines[cont[1]].append(cont)

    valid.append(len(data_lines) == 10)

    while len(data_lines) < 10:
        y_keys = sorted(data_lines.keys())
        diffs = []
        diff_map = {}
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
        print(y_keys)
        print(diffs)
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

        accepted_offset = max(y_ranges, key=lambda p: len(p))

        avg_diff = math.floor(sum(set(accepted_offset)) / len(set(accepted_offset)))
        avg_height = math.floor(sum(set(accepted_range)) / len(set(accepted_range)))
        new_y = data_lines[diff_map[max_diff]][0][1] + data_lines[diff_map[max_diff]][0][3] + avg_diff
        print("PREV_CONTOURS: ", data_lines[diff_map[max_diff]])
        print("NEW Y", new_y)
        for rect in data_lines[diff_map[max_diff]]:
            data_lines[new_y].append([rect[0], new_y, rect[2], avg_height])
        print("NEW_CONTOURS: ", data_lines[new_y])

    final_rects = []
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

    for txt in os.listdir(f"{folder_dir}/text"):
        folder_name = txt.split(".")[0]
        if not os.path.exists(f"{folder_dir}/scorelines/{folder_name}"):
            os.makedirs(f"{folder_dir}/scorelines/{folder_name}")

        for i, final in enumerate(sorted(final_rects, key=lambda x: x[1])):
            x, y, w, h = final
            cv2.rectangle(final_rect_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            pre_crop = clean_edge_cases(text_map[folder_name], f"{folder_dir}/scorelines/{folder_name}/")
            score_line = text_map[folder_name][y + pre_crop:y + h, x:x + w]
            cv2.imwrite(f"{folder_dir}/scorelines/{folder_name}/{i}_score.jpg", score_line)

    cv2.imwrite(f"{folder_dir}/final_rect.jpg", final_rect_image)
    print("Height: ", sorted(h_counter))
    print("Width: ", sorted(w_counter))
    print()
    # print("Excluded Height: ", h_e_counter)
    # print("Excluded Width: ", w_e_counter)
    # print()

print(f"{len(os.listdir('sample/'))} files... {sum(valid)} Valid")
