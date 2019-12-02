import os
from keras.models import Sequential
import pickle
from keras.layers import *
from keras.models import Model
from keras import backend as K
import pandas as pd
from cv2 import imread
from filtering.functions import resize_image
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from filtering.constants import IMAGE_WIDTH, IMAGE_HEIGHT
from tensorflow import compat
from shutil import rmtree
import tensorflow as tf

floatx = "float64"


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    # from tensorflow.compat.v1.graph_util import extract_sub_graph
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def load_label_maps(home_dir="."):
    """

    :return: (labels, LABELS_MAP, BINARY_MAP)
    """
    labels = sorted([label for label in os.listdir("training/") if (label != ".DS_Store")])
    LABEL_MAP = pickle.load(open(f"{home_dir}/label_map.pkl", 'rb'))
    BINARY_MAP = pickle.load(open(f"{home_dir}/binary_map.pkl", 'rb'))
    return labels, LABEL_MAP, BINARY_MAP


def build_and_save_labels():
    labels = sorted([label for label in os.listdir("training/") if (label != ".DS_Store")])
    encoder = LabelBinarizer()
    labels_encoded = encoder.fit_transform(labels)
    LABEL_MAP = {labels[i]: encoded for i, encoded in enumerate(labels_encoded)}
    BINARY_MAP = {"".join([str(int(num)) for num in value]): key for key, value in LABEL_MAP.items()}
    pickle.dump(LABEL_MAP, open("./label_map.pkl", 'wb'))
    pickle.dump(BINARY_MAP, open("./binary_map.pkl", 'wb'))


def get_datasets():
    print("loading datasets")
    training_dataset = []
    testing_dataset = []
    widths = []
    heights = []
    labels, LABEL_MAP, _ = load_label_maps()
    for i, label in enumerate(labels):
        for j, file in enumerate(os.listdir(f"training/{label}")):
            if file != ".DS_Store":
                img = imread(f"training/{label}/{file}", 0)
                img = resize_image(img)
                image_sequence = np.array(file.split(".")[0].split("_")[-1])

                widths.append(img.shape[1])
                heights.append(img.shape[0])

                training_dataset.append((np.array(img), LABEL_MAP[label], image_sequence))

        for k, t_file in enumerate(os.listdir(f"testing/{label}")):
            if t_file != ".DS_Store":
                img = imread(f"testing/{label}/{t_file}", 0)
                img = resize_image(img)
                image_sequence = np.array(t_file.split(".")[0].split("_")[-1])

                widths.append(img.shape[1])
                heights.append(img.shape[0])
                testing_dataset.append((np.array(img), LABEL_MAP[label], image_sequence))

    print(f"Max Height: {max(heights)}")
    print(f"Average Height: {sum(heights) / len(heights)}")
    print(f"Max Width: {max(widths)}")
    print(f"Average Width: {sum(widths) / len(widths)}")
    return testing_dataset, training_dataset


def get_exp_datasets():
    print("Loading experimental dataset")
    training_dataset = []
    testing_dataset = []
    labels, LABEL_MAP, _ = load_label_maps()
    training_df = pd.read_csv("../training_remapped.csv")
    testing_df = pd.read_csv("../testing_remapped.csv")
    for i, label in enumerate(labels):
        for j, file in enumerate(os.listdir(f"experimental/training/{label}")):
            if file != ".DS_Store":
                q = training_df.query(f"""file_name == "{file}" """).query(f"label == '{label}'")
                params = q.iloc(0)[0].to_list()[2:]
                training_dataset.append((np.array(params), LABEL_MAP[label]))

        for k, t_file in enumerate(os.listdir(f"experimental/testing/{label}")):
            if t_file != ".DS_Store":
                q = testing_df.query(f"""file_name == "{t_file}" """).query(f"label == '{label}'")
                params = q.iloc(0)[0].to_list()[2:]
                testing_dataset.append((np.array(params), LABEL_MAP[label]))
    return training_dataset, testing_dataset


def experimental_train():
    K.set_floatx(floatx)

    training_dataset, testing_dataset = get_exp_datasets()
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(6, 1)))
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    training_data = np.stack([np.array(data[0]) for data in training_dataset]).reshape([-1, 6, 1])
    training_labels = np.stack([np.array(data[1]) for data in training_dataset])

    testing_data = np.stack([np.array(data[0]) for data in testing_dataset]).reshape([-1, 6, 1])
    testing_labels = np.stack([np.array(data[1]) for data in testing_dataset])
    print(training_data.shape)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(training_data, training_labels, epochs=10, batch_size=16)
    acc = model.evaluate(testing_data, testing_labels, batch_size=16, verbose=1)
    print(acc)
    model_json = model.to_json()
    with open("exp_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("exp_model.h5")
    print("Saved model to disk")
    return model


def train():
    # force set to float64
    K.set_floatx("float32")
    training_dataset, testing_dataset = get_datasets()
    # model = Sequential()
    inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.float32)  # )
    # conv_1 = Conv2D(256, kernel_size=3, strides=2, activation='relu', )(
    #    inputs)  # input_dtype=tf.float32, dtype=tf.float32

    conv_1 = Conv2D(64, kernel_size=3, strides=1, activation='relu', input_dtype=tf.float32, dtype=tf.float32)(
        inputs)  #
    mp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    do_1 = Dropout(0.2)(conv_1)
    f_1 = Flatten()(do_1)
    dense_1 = Dense(128, activation='relu', dtype=tf.float32)(f_1)  #
    output = Dense(5, activation='softmax', dtype=tf.float32, name="linear/head/predictions/probabilities",
                   input_dtype=tf.float32)(dense_1)  #

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_data = np.array([data[0] for data in training_dataset]).reshape(
        [len(training_dataset), IMAGE_HEIGHT, IMAGE_WIDTH, 1]).astype(
        np.float32)
    training_labels = np.array([data[1] for data in training_dataset])
    tf.cast(training_data, tf.float32)
    tf.cast(training_labels, tf.float32)
    model.fit(training_data, y=training_labels, batch_size=16, epochs=5, )
    # print(training_data.shape)
    # print(training_labels.shape)

    testing_data = np.array([data[0] for data in testing_dataset]).reshape(
        [len(testing_dataset), IMAGE_HEIGHT, IMAGE_WIDTH, 1]).astype(
        np.float32)
    testing_labels = np.array([data[1] for data in testing_dataset])
    tf.cast(testing_data, tf.float32)
    tf.cast(testing_labels, tf.float32)
    acc = model.evaluate(testing_data, testing_labels, batch_size=16, verbose=1)
    print(acc)
    print(model.input)
    # frozen_graph = freeze_session(K.get_session(),
    #                             output_names=[out.op.name for out in model.outputs])
    # tf.io.write_graph(frozen_graph, "../goserver/models/model", "model.pb", as_text=False)

    if os.path.exists("../goserver/models/model"):
        rmtree("../goserver/models/model")
    from filtering.create_go_model import export_model_to_pb
    export_model_to_pb(model, "../goserver/models/model")
    # model.save('model.h5')

    # serialize model to JSON
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #    json_file.write(model_json)
    # serialize weights to HDF5
    # model.save_weights("model.h5")
    # print("Saved model to disk")


train()
# build_and_save_labels()
# train()
# experimental_train()


#####################
# Deprecated Layers #
# image_model()
# model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3), activation='relu'))
#
# model.add(Dense(128, activation='relu'))
# model.add(Conv2D(32, (3, 3), strides=2, activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)))
# model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
# model.add(BatchNormalization())
# model.add(GlobalAveragePooling2D())

# sequence_input = Input(shape=(1,))
# merged = concatenate(inputs=[sequence_input, modelA.output], axis=-1)

# print(merged.shape)
# outputModel = Dense(32, input_shape=(8,), activation='relu')(merged)
# print(outputModel.shape)
# outputModel = Dense(7, activation='softmax')(outputModel)

# model = Model(inputs=[modelA.input, sequence_input], output=outputModel)
# training_sequences = np.array([data[2] for data in training_dataset])
# testing_sequences = np.array([data[2] for data in testing_dataset])

# model.fit(x=[training_data, training_sequences], y=training_labels, batch_size=16, epochs=20, )
# acc = model.evaluate([testing_data, testing_sequences], testing_labels, batch_size=16, verbose=1)

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Experimental
# model.add(Dense(HIDDEN_LAYERS, activation='relu', input_shape=(6,)))
# model.add(Dense(5, activation='softmax'))

# model.add(Conv1D(100, 3, activation='relu', input_shape=(6, 1)))
# model.add(Conv1D(100, 3, activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(Conv1D(160, 3, activation='relu'))
# model.add(Conv1D(160, 3, activation='relu'))
# model.add(GlobalAveragePooling1D())
# model.add(Dropout(0.5))
# model.add(Dense(5, activation='softmax'))
