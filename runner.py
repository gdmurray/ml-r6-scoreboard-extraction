from flask import Flask, render_template, request, jsonify
import keras.backend.tensorflow_backend as tb
from filtering.functions import load_model
from filtering.ml import load_label_maps
from urllib.parse import quote
from sklearn.preprocessing import LabelBinarizer
import redis
import json
import os
from PIL import Image


import base64

tb._SYMBOLIC_SCOPE.value = True
import numpy as np

score_labels, SCORE_LABEL_MAP, SCORE_BINARY_MAP = load_label_maps(home_dir="filtering/")
labels = sorted(["casters", "in_game", "map_ban", "operator_ban", "operator_select", "other", "scoreboard"])
encoder = LabelBinarizer()
binarized = encoder.fit_transform(labels)
LABEL_MAP = {labels[i]: value for i, value in enumerate(binarized)}
BINARY_MAP = {"".join(list(map(str, value))): key for key, value in LABEL_MAP.items()}

DIR = "/home/ubuntu/data"
app = Flask(__name__)
r = redis.Redis(host='localhost')


IMG_WIDTH = int(1280 / 2)
IMG_HEIGHT = int(720 / 2)

RESIZED_WIDTH = int(1280 / 5)
RESIZED_HEIGHT = int(720 / 5)

frame_model = load_model(model_name="model", dir="parser/")
model = load_model()
print("Model Loaded")
exp_model = load_model(model_name="exp_model")
print("Experimental Model Loaded")


@app.route("/ping")
def pong():
    return "pong"



@app.route("/predict")
def get_scoreboards():
    image_b64 = base64.urlsafe_b64decode(request.args.get('image'))
    exp_b64 = base64.urlsafe_b64decode(request.args.get("exp"))
    print(len(image_b64))
    model_array = np.frombuffer(image_b64, dtype=np.float64).reshape(50, 150)
    model_array = model_array.reshape(-1, 50, 150, 1)

    exp_array = np.frombuffer(exp_b64, dtype=np.float64)
    exp_array = exp_array.reshape([-1, 6, 1])

    model_prediction = model.predict(model_array)
    exp_model_prediction = exp_model.predict(exp_array)

    exp_prediction_string = "".join(['1' if (m == np.argmax(exp_model_prediction[0])) else '0' for m, num in
                                     enumerate(exp_model_prediction[0])])
    exp_prediction = SCORE_BINARY_MAP[exp_prediction_string]

    # prediction = model.predict([processed_img, np.array([i])])
    prediction_string = "".join([str(int(round(num, 0))) for num in model_prediction[0]])
    if prediction_string == "00000":
        prediction_string = "".join(['1' if (m == np.argmax(model_prediction[0])) else '0' for m, num in
                                     enumerate(model_prediction[0])])

    prediction = SCORE_BINARY_MAP[prediction_string]
    return jsonify({"prediction": str(prediction), "exp_prediction": str(exp_prediction)})


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
            prediction = frame_model.predict(img_array)
            pred = [int(round(m, 0)) for m in prediction[0]]
            key = "".join(list(map(str, pred)))
        if key == "0000000":
            key = "".join(['1' if (m == np.argmax(prediction[0])) else '0' for m, num in
                           enumerate(prediction[0])])
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
