from tensorflow import keras
from tensorflow.python.keras import layers
import tensorflow as tf
import os

# from layers import BatchNormalization
# layers.BatchNormalization = BatchNormalization

IMG_SIZE = 71
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


def load_inception_v3(load_latest_checkpoint=True):
    weights = None if load_latest_checkpoint else 'imagenet'

    base_model = keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights=weights)
    base_model.trainable = True

    # layer_after_mixed7_index = None
    # for index, layer in enumerate(base_model.layers):
    #     if layer.name == 'mixed7':
    #         layer_after_mixed7_index = index + 1
    #         break
    #
    # for layer in base_model.layers[:layer_after_mixed7_index]:
    #     layer.trainable = False
    #
    # for layer in base_model.layers[layer_after_mixed7_index:]:
    #     layer.trainable = True

    global_average_pooling_2d_layer = layers.GlobalAveragePooling2D()
    dense_layer1 = layers.Dense(1024, activation='relu')
    dense_layer2 = layers.Dense(1024, activation='relu')
    prediction_layer = layers.Dense(10, activation='softmax')

    model = keras.Sequential([
        base_model,
        global_average_pooling_2d_layer,
        dense_layer1,
        dense_layer2,
        prediction_layer
    ])

    if load_latest_checkpoint:
        __load_latest_checkpoint(model)

    return model


def load_mobile_net_v2(load_latest_checkpoint=True):
    weights = None if load_latest_checkpoint else 'imagenet'

    base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights=weights)
    base_model.trainable = True

    flatten_layer = layers.Flatten()
    dense_layer1 = layers.Dense(1024, activation='relu')
    dense_layer2 = layers.Dense(1024, activation='relu')
    prediction_layer = layers.Dense(10, activation='softmax')

    model = keras.Sequential([
        base_model,
        flatten_layer,
        dense_layer1,
        dense_layer2,
        prediction_layer
    ])

    if load_latest_checkpoint:
        __load_latest_checkpoint(model)

    return model


def load_res_net_50(load_latest_checkpoint=True):
    weights = None if load_latest_checkpoint else 'imagenet'

    base_model = keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights=weights)
    base_model.trainable = True

    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(1024, activation='relu')
    prediction_layer = layers.Dense(10, activation='softmax')

    model = keras.Sequential([
        base_model,
        flatten_layer,
        dense_layer,
        prediction_layer
    ])

    if load_latest_checkpoint:
        __load_latest_checkpoint(model)

    return model


def load_xception(load_latest_checkpoint=True):
    weights = None if load_latest_checkpoint else 'imagenet'

    base_model = keras.applications.Xception(input_shape=IMG_SHAPE, include_top=False, weights=weights)
    base_model.trainable = True

    global_average_pooling_2d_layer = layers.GlobalAveragePooling2D()
    dense_layer1 = layers.Dense(512, activation='relu')
    dense_layer2 = layers.Dense(512, activation='relu')
    prediction_layer = layers.Dense(10, activation='softmax')

    model = keras.Sequential([
        base_model,
        global_average_pooling_2d_layer,
        dense_layer1,
        dense_layer2,
        prediction_layer
    ])

    if load_latest_checkpoint:
        __load_latest_checkpoint(model)

    return model


def __load_latest_checkpoint(model):
    checkpoint_dir = os.path.join('model')
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest_checkpoint)
