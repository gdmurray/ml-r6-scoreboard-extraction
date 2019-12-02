import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import model_from_json
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
from random import shuffle
import os
import time

DIR = "D://data/training_sets/"
TEST_DIR = "D://data/testing_set/"
CACHE_DIR = "D://data/cache/"
IMG_WIDTH = int(1280 / 2)
IMG_HEIGHT = int(720 / 2)
values = os.listdir(DIR)
encoder = LabelBinarizer()
labels = ["casters", "in_game", "map_ban", "operator_ban", "operator_select", "other", "scoreboard"]

binarized = encoder.fit_transform(labels)
LABEL_MAP = {values[i]: value for i, value in enumerate(binarized)}
BINARY_MAP = {"".join(list(map(str, value))): key for key, value in LABEL_MAP.items()}


def get_height_avgs():
    heights = []
    widths = []

    for folder in os.listdir(DIR):
        folder_dir = f"{DIR}{folder}"
        files = os.listdir(folder_dir)
        print(f"Processing: {folder}... {len(files)} images")
        start = time.time()
        for img in files:
            with Image.open(f"{folder_dir}/{img}") as image:
                data = np.array(image)
                heights.append(data.shape[0])
                widths.append(data.shape[1])
        finish = time.time()
        print(f"Took {finish - start:.2f}s")
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)

    print(f"[Height] avg: {avg_height}, max: {max(heights)}, min: {min(heights)}")
    print(f"[Width] avg: {avg_width}, max: {max(avg_width)}, min: {min(avg_width)}")


def load_testing_data():
    print(f"Loading Testing Data from: {TEST_DIR}")
    test_data = []
    for folder in os.listdir(TEST_DIR):
        folder_dir = f"{TEST_DIR}{folder}"
        files = os.listdir(folder_dir)
        print(f"Processing: {folder}... {len(files)} images")
        start = time.time()
        for img_file in files:
            with Image.open(f"{folder_dir}/{img_file}") as image:
                img = image.convert('L')
                img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)
                test_data.append([np.array(img), LABEL_MAP[folder]])
        finish = time.time()
        print(f"Took {finish - start:.2f}s")
    shuffle(test_data)
    return test_data


def load_training_data():
    train_data = []
    for folder in os.listdir(DIR):
        folder_dir = f"{DIR}{folder}"
        files = os.listdir(folder_dir)
        print(f"Processing: {folder}... {len(files)} images")
        start = time.time()
        for img_file in files:
            with Image.open(f"{folder_dir}/{img_file}") as image:
                img = image.convert('L')
                img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)
                train_data.append([np.array(img), LABEL_MAP[folder]])
        finish = time.time()
        print(f"Took {finish - start:.2f}s")
    shuffle(train_data)
    return train_data


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


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


def test_model():
    model = load_model()
    test_data = load_testing_data()
    test_images = np.array([i[0] for i in test_data]).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    test_labels = np.array([i[1] for i in test_data])
    print(test_labels)
    print(test_images.shape[0], test_images.shape[1])
    print(test_labels.shape[0], test_labels.shape[1])
    scores = model.evaluate(test_images, test_labels, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


def train():
    print("Loading Training Data")
    train_data = load_training_data()
    train_images = np.array([i[0] for i in train_data]).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    train_labels = np.array([i[1] for i in train_data])
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_images, train_labels, batch_size=16, epochs=5, verbose=1)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    print("Testing Model")
    test_data = load_testing_data()
    test_images = np.array([i[0] for i in test_data]).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    test_labels = np.array([i[1] for i in test_images])

    scores = model.evaluate(test_images, test_labels, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
