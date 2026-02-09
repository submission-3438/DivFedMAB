import tensorflow as tf
from tensorflow.keras import layers, models

def conv3x3(filters, stride=1):
    return layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)

def basic_block(x, filters, stride=1):
    shortcut = x

    x = conv3x3(filters, stride)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = conv3x3(filters, 1)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def make_layer(x, filters, blocks, stride):
    x = basic_block(x, filters, stride)
    for _ in range(1, blocks):
        x = basic_block(x, filters)
    return x

def create_model(num_classes=10):
    inputs = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = make_layer(x, 64, 2, stride=1)
    x = make_layer(x, 128, 2, stride=2)
    x = make_layer(x, 256, 2, stride=2)
    x = make_layer(x, 512, 2, stride=2)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
