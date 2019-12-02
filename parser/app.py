from flask import Flask, render_template
import keras.backend.tensorflow_backend as tb
from keras.models import model_from_json
from urllib.parse import quote
from sklearn.preprocessing import LabelBinarizer
import redis
from PIL import Image

import base64

tb._SYMBOLIC_SCOPE.value = True
from PIL import Image
import os
import numpy as np

DIR = "/home/ubuntu/data"
app = Flask(__name__)
r = redis.Redis(host='localhost')

IMG_WIDTH = int(1280 / 2)
IMG_HEIGHT = int(720 / 2)

RESIZED_WIDTH = int(1280 / 5)
RESIZED_HEIGHT = int(720 / 5)

labels = sorted(["casters", "in_game", "map_ban", "operator_ban", "operator_select", "other", "scoreboard"])
encoder = LabelBinarizer()
binarized = encoder.fit_transform(labels)
LABEL_MAP = {labels[i]: value for i, value in enumerate(binarized)}
BINARY_MAP = {"".join(list(map(str, value))): key for key, value in LABEL_MAP.items()}


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    print("Compiling model")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


model = load_model()
print("Model Loaded")


@app.route("/ping")
def pong():
    return "pong"


@app.route("/")
def main():
    data = {"_".join([c for c in d.split(" ")]): d for d in os.listdir(DIR)}
    return render_template("main.html", data=data)


@app.route("/data/<folder>")
def data_view(folder):
    folder_ = folder
    folder = " ".join([c for c in folder.split("_")])
    if not os.path.exists(f"{DIR}/{folder}/thumbnail"):
        os.mkdir(f"{DIR}/{folder}/thumbnail")
    items = os.listdir(f"{DIR}/{folder}")
    return render_template("data.html", folder_=folder_, folder=folder, num_items=len(items))


@app.route("/predict/<folder>/scoreboards")
def get_scoreboards(folder):
    pass


@app.route("/predict/<folder>/<image>")
def get_prediction(folder, image):
    cache_name = f"cache/{folder}"

    folder = " ".join([c for c in folder.split("_")])
    image_list = os.listdir(f"{DIR}/{folder}")
    video_id = image_list[0].split("_")[1]

    image_name = f"frame_{video_id}_{image}.jpg"
    if not os.path.isfile(f"{DIR}/{folder}/thumbnail/{image_name}"):
        img = Image.open(f"{DIR}/{folder}/{image_name}")
        img = img.resize((RESIZED_WIDTH, RESIZED_HEIGHT), Image.ANTIALIAS)
        img.save(f"{DIR}/{folder}/thumbnail/{image_name}")

    if image != len(image_list) - 1:
        next_image = int(image) + 1
    else:
        next_image = None

    image_url = f"http://52.72.134.139/static/{quote(folder)}/thumbnail/{image_name}"

    cache_key = f"{cache_name}/{image.split('.')[0]}"
    is_cached = False
    use_redis = True
    try:
        if r.exists(cache_key):
            final_prediction = r.get(cache_key).decode("utf-8")
            is_cached = True
    except redis.exceptions.ConnectionError:
        use_redis = False

    if not is_cached:
        image_path = f"{DIR}/{folder}/{image_name}"
        with Image.open(image_path) as img_f:
            img = img_f.convert('L')
            img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)
            img_array = np.array(img).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
            prediction = model.predict(img_array)
            pred = [int(round(m, 0)) for m in prediction[0]]
            key = "".join(list(map(str, pred)))
        final_prediction = BINARY_MAP[key]
        if use_redis:
            r.set(cache_key, final_prediction)
    data = {
        "image_path": image_url,
        "image_name": image_name,
        "prediction": final_prediction,
        "image_num": image,
        "next": next_image
    }
    return data


if __name__ == "__main__":
    app.run(debug=False, threaded=False)
