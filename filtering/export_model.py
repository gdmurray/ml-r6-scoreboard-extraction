from keras import backend as K
from keras.models import load_model, model_from_json
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf


def export_model_to_pb(keras_model, export_path):
    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(
        inputs={"image": tf.placeholder(tf.float32, keras_model.input.shape, name="input")},
        outputs={"score": keras_model.output})

    with K.get_session() as sess:
        # Save the meta graph and the variables
        builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                             signature_def_map={"predict": signature})

    builder.save()


def export_h5_to_pb(model_name, export_path):
    K.set_learning_phase(0)
    K.set_floatx("float64")

    keras_model = load_model(f"{model_name}.h5")

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={"images": keras_model.input},
                                      outputs={"scores": keras_model.output})

    with K.get_session() as sess:
        # Save the meta graph and the variables
        builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                             signature_def_map={"predict": signature})

    builder.save()
