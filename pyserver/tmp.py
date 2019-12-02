def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def resize_image(img):
    if img.shape[1] > IMAGE_WIDTH:
        img = image_resize(img, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

    if img.shape[0] < IMAGE_HEIGHT and img.shape[1] < IMAGE_WIDTH:
        y = int(math.ceil((IMAGE_HEIGHT - img.shape[0]) / 2))
        x = int(math.ceil((IMAGE_WIDTH - img.shape[1]) / 2))
        img = cv2.copyMakeBorder(img, y, y, x, x, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    if img.shape[0] < IMAGE_WIDTH:
        y = int(math.ceil((IMAGE_HEIGHT - img.shape[0]) / 2))
        img = cv2.copyMakeBorder(img, y, y, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # if img.shape[0] > 50:
    #    img = cv2.resize()
    # if (f.shape[1] < 50):
    # top, bottom, left, right, borderType
    #    constant = cv2.copyMakeBorder(f, 15, 16, 10, 11, cv2.BORDER_CONSTANT, value=BLUE)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    return img


@app.route("/predict")
def predict():
    print(os.listdir('data/'))
    image_uuid = uuid.uuid1()
    # Looping through SCORES
    master_start = time.time()
    times = []
    count = 0
    errors = []
    for file in os.listdir("/code/data/"):
        file_name = file.split(".")[0]
        img = cv2.imread(f"/code/data/{file}", 0)
        print(img)
        x_pixels = img.shape[1] // 9
        # looping through BOUNDING BOXES
        for x_idx in range(0, 9):
            sub_img = np.array(img[0:img.shape[0], x_idx * x_pixels:(x_idx * x_pixels) + x_pixels])
            # retval, buffer = cv2.imencode('.jpg', sub_img)

            # sub_img_bytes = np.array(buffer)
            resized = np.array(resize_image(sub_img))
            key = f"{image_uuid}/{file_name}/text_{x_idx}/predict"
            r.set(key, resized.tobytes())
            start = time.time()
            response = requests.get(f"http://goserver:8080/predict/{image_uuid}/{file_name}/text_{x_idx}")
            count += 1
            print(response.json())
            finish = time.time()
            times.append(finish - start)

    print()
    master_finish = time.time()
    print(f"Took {master_finish - master_start:.2f}s to do {count} requests")
    return {"requests": count,
            "total": f"{master_finish - master_start:.2f}",
            "min": f"{min(times):.2f}s",
            "max": f"{max(times):.2f}s",
            "avg": f"{sum(times) / len(times):.2f}s"}


@app.route('/ocr')
def hello():
    print(os.listdir('data/'))
    image_uuid = "TESTING_IMAGE"
    # Looping through SCORES
    master_start = time.time()
    times = []
    count = 0
    errors = []
    for file in os.listdir("/code/data/"):
        file_name = file.split(".")[0]
        img = cv2.imread(f"/code/data/{file}", 0)
        print(img)
        x_pixels = img.shape[1] // 9
        # looping through BOUNDING BOXES
        for x_idx in range(0, 9):
            sub_img = np.array(img[0:img.shape[0], x_idx * x_pixels:(x_idx * x_pixels) + x_pixels])
            par_key = f"{image_uuid}/{file_name}/text_{x_idx}"
            # retval, buffer = cv2.imencode('.jpg', sub_img)
            b64_query_string = ""
            retval, buffer = cv2.imencode('.jpg', sub_img)
            jpg_as_text = base64.urlsafe_b64encode(buffer)
            for text in ["text1", "text2", "text3", "text4"]:
                b64_query_string += f"&{text}={jpg_as_text}"

            start = time.time()
            choices = ["text", "number"]
            img_type = choices[random.randint(0, 1)]
            query_string = f"?type={img_type}&x={sub_img.shape[1]}&y={sub_img.shape[0]}{b64_query_string}"
            response = requests.get(
                f"http://goserver:8080/ocr/{image_uuid}/{file_name}/text_{x_idx}{query_string}"
            )
            count += 1
            print(response.json())
            finish = time.time()
            times.append(finish - start)

    print()
    master_finish = time.time()
    print(f"Took {master_finish - master_start:.2f}s to do {count} requests")
    return {"requests": count,
            "total": f"{master_finish - master_start:.2f}",
            "min": f"{min(times):.2f}s",
            "max": f"{max(times):.2f}s",
            "avg": f"{sum(times) / len(times):.2f}s"}