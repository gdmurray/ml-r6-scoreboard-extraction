from flask import Flask, render_template
import keras.backend.tensorflow_backend as tb
from keras.models import model_from_json
from urllib.parse import quote
from sklearn.preprocessing import LabelBinarizer
from scoreboard.processor import Processor
import redis
import cv2
from PIL import Image
from celery import Celery

import base64

tb._SYMBOLIC_SCOPE.value = True
from PIL import Image
import os
import numpy as np

DIR = "/home/ubuntu/data"
app = Flask(__name__)
r = redis.Redis(host='localhost')

app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

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
    json_file = open('filtering/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("filtering/model.h5")
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


@app.route("/<folder>/scoreboard/<image>")
def get_scoreboard(folder, image):
    folder_ = folder
    cache_name = f"cache/{folder}"
    folder = " ".join([c for c in folder.split("_")])

    image_path = f"http://52.72.134.139/static/{quote(folder)}/{image}"
    process_image.apply_async(args=[f"/home/ubuntu/data/{folder}/{image}", image])
    return render_template("scoreboard.html", image=image, image_path=image_path)


@celery.task
def process_image(path, image):
    file = cv2.imread(path, 0)
    processor = Processor(file, image, cache=True)
    img = processor.extract_scoreboard_box(file)
    processor.extract_scorelines(img)
    processor.extract_scores()
    return


@app.route("/<folder>/scoreboards")
def get_scoreboards(folder):
    folder_ = folder
    cache_name = f"cache/{folder}"
    folder = " ".join([c for c in folder.split("_")])
    image_list = os.listdir(f"{DIR}/{folder}")
    scoreboard_images = []
    from collections import Counter
    c = Counter()
    images = []
    for image in image_list:
        image_name = image
        image_number = image.split(".")[0].split("_")[-1]
        cache_key = f"{cache_name}/{image_number}"
        prediction = r.get(cache_key)
        if prediction:
            prediction = prediction.decode("utf-8")
            c[prediction] += 1
            if prediction == "scoreboard":
                images.append((image, f"http://52.72.134.139/static/{quote(folder)}/{image_name}",
                               f"/{folder_}/scoreboard/{image_name}"))

    return render_template("scoreboards.html", images=images, folder_=folder_)


if __name__ == "__main__":
    app.run(debug=False, threaded=False)
