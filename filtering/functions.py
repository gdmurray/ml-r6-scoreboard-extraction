from keras.models import model_from_json
import math
import cv2

IMAGE_HEIGHT = 50
IMAGE_WIDTH = 150

def load_model(model_name="model", dir="filtering/"):
    print(f"Loading model: {model_name}")
    json_file = open(f"{dir}{model_name}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f"{dir}{model_name}.h5")
    print("Loaded model from disk... Compiling")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model



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
