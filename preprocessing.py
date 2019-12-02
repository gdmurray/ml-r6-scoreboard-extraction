import cv2
import os
import numpy as np
import math

for file in os.listdir("sample/"):
    # Read Image
    print(file)
    img = cv2.imread(f"sample/{file}", 0)
    folder_dir = f"out/{file.split('.')[0]}"

    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    th1 = 255 - th1
    cv2.imwrite(f"{folder_dir}/out.jpg", th1)

    kernel_length = np.array(img).shape[1] // 80

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_temp1 = cv2.erode(th1, v_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(img_temp1, v_kernel, iterations=15)
    cv2.imwrite(f"{folder_dir}/verticle_lines.jpg", vertical_lines_img)

    img_temp2 = cv2.erode(th1, h_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, h_kernel, iterations=15)
    cv2.imwrite(f"{folder_dir}/horizontal_lines.jpg", horizontal_lines_img)

    alpha = 0.5
    beta = 1.0 - alpha
    img_final_bin = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=1)
    cv2.imwrite(f"{folder_dir}/img_pre_thresh.jpg", img_final_bin)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(f"{folder_dir}/img_final_bin.jpg", img_final_bin)

    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    empty = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    empty.fill(255)

    i = 0
    for c in contours:
        i += 1
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        # a, b, w2, h2 = cv2.boundingRect(c)

        if x > 100 and ((x + w) < img.shape[1]) and y > 100:
            cv2.rectangle(empty, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(f"{folder_dir}/approx.jpg", empty)

    kernel_length_p2 = np.array(img).shape[1] // 80
    v_kernel_p2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_p2))
    h_kernel_p2 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_p2, 1))
    kernel_p2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_temp1_pass2 = cv2.erode(empty, v_kernel_p2, iterations=3)
    v_lines_img = cv2.dilate(img_temp1_pass2, v_kernel_p2, iterations=3)

    img_temp2_pass2 = cv2.erode(empty, h_kernel_p2, iterations=3)
    h_lines_img = cv2.dilate(img_temp2_pass2, h_kernel_p2, iterations=3)

    alpha = 0.5
    beta = 1.0 - alpha
    img_final_bin_pass2 = cv2.addWeighted(v_lines_img, alpha, h_lines_img, beta, 0.0)
    img_final_bin_pass2 = cv2.erode(~img_final_bin_pass2, kernel_p2, iterations=1)
    img_final_bin_pass2 = cv2.dilate(img_final_bin_pass2, kernel_p2, iterations=2)
    # img_final_bin_pass2 = cv2.erode(img_final_bin_pass2, kernel_p2, iterations=1)
    (thresh, img_final_bin_pass2) = cv2.threshold(img_final_bin_pass2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(f"{folder_dir}/img_final_bin_p2.jpg", img_final_bin_pass2)

    zero_corners = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    zero_corners.fill(255)

    contours_p2, hierarchy_p2 = cv2.findContours(img_final_bin_pass2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edges = cv2.Canny(img_final_bin_pass2, 100, 200, apertureSize=3)

    corners = cv2.goodFeaturesToTrack(edges, 30, 0.01, 100)

    corner_points = [(c.ravel()[0], c.ravel()[1]) for c in corners]
    corners = np.int0(corners)
    for corn in corners:
        x, y = corn.ravel()
        cv2.circle(zero_corners, (x, y), 5, (0, 255, 0), -1)

    cv2.imwrite(f"{folder_dir}/corners.jpg", zero_corners)
    cv2.imwrite(f"{folder_dir}/edges.jpg", edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)

    # print(f"Hough Lines: {len(lines)}")
    horizontal_lines = []
    vertical_lines = []

    # def point_intersects()
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

            # Cut off Slanted Lines
            if abs_x <= 1 and abs_y <= 1:
                if a < 0:
                    h_check = [(abs(c[1] - y2) < 3) for c in corner_points]
                    if any(h_check):
                        cv2.line(zero_corners, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        horizontal_lines.append([(x1, y1), (x2, y2)])
                else:
                    v_check = [(abs(c[0] - x2) < 5) for c in corner_points]
                    if any(v_check):
                        cv2.line(zero_corners, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        vertical_lines.append([(x1, y1), (x2, y2)])

    cv2.imwrite(f"{folder_dir}/houghlines.jpg", zero_corners)

    bounding_box = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    bounding_box.fill(255)

    if len(contours_p2) > 1:
        print("More than 1 Contour p2")
        contour_p2 = max(contours_p2, key=lambda x: cv2.boundingRect(x)[2] * cv2.boundingRect(x)[3])
    else:
        contour_p2 = contours_p2[0]
    print(len(contours_p2))
    peri = cv2.arcLength(contour_p2, True)
    approx = cv2.approxPolyDP(contour_p2, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)

    cv2.rectangle(bounding_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(f"{folder_dir}/bounding_box.jpg", bounding_box)

    # Corner Line check to restrict Bounds
    sorted_horizontal_lines = sorted(horizontal_lines, key=lambda x: x[1][1])
    sorted_vertical_lines = sorted(vertical_lines, key=lambda x: x[1][0])

    uppermost_horizontal_line = sorted_horizontal_lines[0]
    lowermost_horizontal_line = sorted_horizontal_lines[-1]

    leftmost_vertical_line = sorted_vertical_lines[0]
    rightmost_vertical_line = sorted_vertical_lines[-1]

    print(uppermost_horizontal_line)
    print(lowermost_horizontal_line)
    print(leftmost_vertical_line)
    print(rightmost_vertical_line)
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
    final_bounding = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    final_bounding.fill(255)
    cv2.rectangle(final_bounding, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(f"{folder_dir}/final_bounds.jpg", final_bounding)
    cropped = img[y:y + h, x:x + w]
    cv2.imwrite(f"{folder_dir}/final_scoreboard.jpg", cropped)
