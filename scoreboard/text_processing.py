from pytesseract import image_to_string, image_to_data
import os
import cv2
from PIL import Image
from collections import Counter
import time
import json
import copy
import numpy as np
import csv
import pandas as pd
from pandas.api.types import is_numeric_dtype
from funcs import flatten
from filtering.functions import resize_image  # load_model
from filtering.constants import IMAGE_WIDTH, IMAGE_HEIGHT
import concurrent.futures
import pickle


def get_exp_params(x, y, w, h, file_, sub_img, seq):
    file_h, file_w = file_.shape

    pixels = sub_img.shape[0] * sub_img.shape[1]
    non_zero = cv2.countNonZero(sub_img)
    ratio = 1 - (non_zero / pixels)

    x_ratio = (x / file_w)
    w_ratio = (w / file_w)
    y_ratio = (y / file_h)
    h_ratio = (y / file_h)
    return np.array([x_ratio, y_ratio, w_ratio, h_ratio, ratio, seq]).reshape([-1, 6, 1])


for file in ['capture2.jpg/']:  # os.listdir("sample/"):   #   # os.listdir("sample/"):  # ['frame_467.jpg']:
    file_name = file.split('.')[0]
    folder_dir = f"proc_out/{file_name}"
    out_dir = f"text_processing/"
    print(file)
    if not os.path.exists(f"text_processing/{file_name}"):
        os.makedirs(f"text_processing/{file_name}")
    # Bounding Box Text processing => text2 => cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    # Text Extraction => text4? attempt
    BB_TEXT = "text2"
    EXTRACT_TEXT = "text4"
    total_start = time.time()
    prediction_dict = {
        "file": file_name,
        "scores": []
    }


    # for score_idx, score_img in enumerate(sorted(os.listdir(f"{folder_dir}/scorelines/text2"))):

    def process_scoreline(score_idx):
        # model = load_model()
        # exp_model = load_model(model_name="exp_model")

        LABEL_MAP = pickle.load(open("filtering/label_map.pkl", 'rb'))
        BINARY_MAP = pickle.load(open("filtering/binary_map.pkl", 'rb'))

        score_img = sorted(os.listdir(f"{folder_dir}/scorelines/text2"))[score_idx]
        print(f"Processing Score index: {score_idx}... {score_img}")
        txt_list = ['text1', 'text2', 'text3', 'text4']
        text_img_map = {txt: cv2.imread(f"{folder_dir}/scorelines/{txt}/{score_img}", 0) for txt in txt_list}

        start = time.time()
        score_name = score_img.split(".")[0]
        img_dir = f"{folder_dir}/scorelines/{BB_TEXT}/{score_img}"
        img = cv2.imread(img_dir, 0)

        empty_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        empty_img.fill(255)

        img = 255 - img

        cv2.imwrite(f"text_processing/{file_name}/original_{score_img}", img)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilation = cv2.dilate(img, rect_kernel, iterations=1)
        cv2.imwrite(f"text_processing/{file_name}/dilation_{score_img}", dilation)

        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        crop_w = 0
        crop_y = 0
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
            for x in range(img.shape[1]):
                for y in range(img.shape[0]):
                    if img[y, x] < 2:
                        img[y, x] = 0
                    if img[y, x] > 253:
                        img[y, x] = 255

            cv2.imwrite(f"text_processing/{file_name}/original_{score_img}", img)
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            new_dilation = cv2.dilate(img, rect_kernel, iterations=1)
            cv2.imwrite(f"text_processing/{file_name}/dilation_{score_img}", new_dilation)

            contours, hierarchy = cv2.findContours(new_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print("New Contours: ", len(contours))

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(empty_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        cv2.imwrite(f"text_processing/{file_name}/contours_{score_img}", empty_img)

        if not os.path.exists(f"text_processing/{file_name}/{score_name}"):
            os.makedirs(f"text_processing/{file_name}/{score_name}")

        text_img = cv2.imread(f"{folder_dir}/scorelines/{EXTRACT_TEXT}/{score_img}")

        if crop_y != 0 or crop_w != 0:
            print("cropxy")
            for tx in text_img_map:
                tmp_img = text_img_map[tx]
                text_img_map[tx] = tmp_img[0 + crop_y:tmp_img.shape[0], 0 + crop_w:tmp_img.shape[1]]

        i = 1
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
        cv2.imwrite(f"text_processing/{file_name}/{score_name}/original.jpg", text_img)

        rows = [headers, ]
        conf_table = [headers[:-2], ]
        of = open(f"text_processing/{file_name}/{score_name}/text.csv", 'w')
        cf = open(f"text_processing/{file_name}/{score_name}/conf.csv", 'w')
        writer = csv.writer(of)
        conf_writer = csv.writer(cf)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # width_counter[w] += 1
            # area_counter[w * h] += 1
            if w > 8:
                rows.append([i, ])
                conf_table.append([i, ])
                text2_img_og = text_img_map['text2']
                text2_img = text2_img_og[y:y + h, x:x + w]
                rsz = resize_image(text2_img)
                np.savetxt("out2.txt", rsz)
                processed_img = np.array(rsz).reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

                print(processed_img[0])

                print(processed_img)
                print(processed_img.shape)
                # model_prediction = model.predict(processed_img)
                model_prediction = [[0, 0, 1, 0, 0]]

                # exp_model_prediction = exp_model.predict(get_exp_params(x, y, w, h, text2_img_og, text2_img, i))
                exp_model_prediction = [[0, 0, 1, 0, 0]]
                exp_prediction_string = "".join(['1' if (m == np.argmax(exp_model_prediction[0])) else '0' for m, num in
                                                 enumerate(exp_model_prediction[0])])
                exp_prediction = BINARY_MAP[exp_prediction_string]

                # prediction = model.predict([processed_img, np.array([i])])
                prediction_string = "".join([str(int(round(num, 0))) for num in model_prediction[0]])
                if prediction_string == "00000":
                    prediction_string = "".join(['1' if (m == np.argmax(model_prediction[0])) else '0' for m, num in
                                                 enumerate(model_prediction[0])])

                prediction = BINARY_MAP[prediction_string]
                # print(f"prediction: {BINARY_MAP[prediction_string]}, {exp_prediction}")

                box_crop = 0
                if prediction == "name" and i < 3:
                    name_img = copy.deepcopy(text2_img)
                    thresh = cv2.threshold(name_img, 160, 255, cv2.THRESH_BINARY_INV)[1]
                    d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    name_dilation = cv2.dilate(thresh, rect_kernel, iterations=2)
                    b_contours, b_hierarchy = cv2.findContours(name_dilation, cv2.RETR_EXTERNAL,
                                                               cv2.CHAIN_APPROX_NONE)
                    c_img = np.zeros([text2_img.shape[0], text2_img.shape[1]], dtype=np.uint8)
                    c_img.fill(255)
                    name_contours = [cv2.boundingRect(b_cont) for b_cont in b_contours]
                    box_contour = min(name_contours, key=lambda box: abs(box[2] - box[3]))
                    box_crop = box_contour[0] + box_contour[2]
                    cv2.imwrite(f"text_processing/{file_name}/{score_name}/name_contours.jpg", c_img)

                for t in txt_list:
                    t_img = text_img_map[t]
                    bound_img = t_img[y:y + h, x + box_crop:x + w]
                    if i >= 3 or prediction in ['number']:
                        data = image_to_data(bound_img, lang="eng_best", config='--psm 10 --oem 3',
                                             output_type='dict')
                    else:
                        data = image_to_data(bound_img, lang="eng_best", config='--psm 12', output_type='dict')
                    dataframe = pd.DataFrame(data)
                    # print(dataframe)
                    dataframe['conf'] = dataframe['conf'].astype(int)
                    lines = dataframe.groupby('block_num')['text'].apply(list)
                    line = max(lines, key=lambda l: len("".join([s for s in l if s != ''])))
                    data_string = ''.join([l for l in line])
                    block_lines = {"".join([s for s in value if s != '']): key for key, value in lines.items()}
                    block_num = block_lines[data_string]
                    conf_df = dataframe.groupby(['block_num', 'conf'])['text'].apply(list)
                    # print(conf_df)
                    conf_keys = [k for k in conf_df[block_num].keys() if k != -1]
                    conf_string = "".join(flatten([conf_df[block_num][k] for k in conf_keys]))
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
                cv2.imwrite(f"text_processing/{file_name}/{score_name}/text_{file_name}_{score_idx}_{i}.jpg",
                            text2_img)
                i += 1

        writer.writerows(rows)
        conf_writer.writerows(conf_table)
        finish = time.time()

        score_dict = {
            "order": score_idx,
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
            score_dict[key]["alternatives"] = [score_dict[key]["alternatives"][p] for p in
                                               score_dict[key]["alternatives"] if
                                               str(score_dict[key]["alternatives"][p][0]) != str(
                                                   score_dict[key]["value"][0])]
        print(f"{score_name} took {finish - start:.2f}s")
        return score_dict


    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        args = (score_idx for score_idx in range(0, len(os.listdir(f"{folder_dir}/scorelines/text2"))))
        for result in executor.map(process_scoreline, args):
            print(result)

    # json.dump(prediction_dict, open(f"text_processing/{file_name}/scores.json", "w"))
    total_finish = time.time()
    print(f"{file_name} took {total_finish - total_start:.2f}s")
