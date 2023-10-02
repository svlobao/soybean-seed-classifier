import os
import cv2
import random
import numpy as np
import tensorflow as tf

from enum import Enum
from sklearn.model_selection import train_test_split


class SplitSize(Enum):
    TRAIN = 0.7
    VAL = 0.1
    TEST = 0.2


class Classes(Enum):
    BROKEN = "broken"
    IMMATURE = "immature"
    INTACT = "intact"
    SKIN_DAMAGED = "skin-damaged"
    SPOTTED = "spotted"


class ImageShape(Enum):
    WIDTH = 227  # columns
    HEIGHT = 227  # lines
    CHANNELS = 3


def loadImageFiles(img_dirs: dict[str, str]) -> dict[str, list[str]]:
    """
    Loads images for all classes based on one input:

    1. A dictionary containing the paths to the folders containing the images of each class. The input dict should come in the following structure:

    \t{
        \t"class_1": StrPath,\n
        \t"class_2": StrPath,\n
        \t"class_3": StrPath,\n
        \t...\n
        \t"class_n": StrPath,\n
    \t}

    """
    broken = [
        os.path.join(img_dirs[Classes.BROKEN.value], img_file)
        for img_file in os.listdir(img_dirs[Classes.BROKEN.value])
    ]

    immature = [
        os.path.join(img_dirs[Classes.IMMATURE.value], img_file)
        for img_file in os.listdir(img_dirs[Classes.IMMATURE.value])
    ]

    intact = [
        os.path.join(img_dirs[Classes.INTACT.value], img_file)
        for img_file in os.listdir(img_dirs[Classes.INTACT.value])
    ]

    skin_damaged = [
        os.path.join(img_dirs[Classes.SKIN_DAMAGED.value], img_file)
        for img_file in os.listdir(img_dirs[Classes.SKIN_DAMAGED.value])
    ]

    spotted = [
        os.path.join(img_dirs[Classes.SPOTTED.value], img_file)
        for img_file in os.listdir(img_dirs[Classes.SPOTTED.value])
    ]

    return {
        Classes.BROKEN.value: broken,
        Classes.IMMATURE.value: immature,
        Classes.INTACT.value: intact,
        Classes.SKIN_DAMAGED.value: skin_damaged,
        Classes.SPOTTED.value: spotted,
    }


def splitDatasets(sample_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    all_classes_split = {}

    for key, value in sample_dict.items():
        train_files, val_test_files = train_test_split(
            value,
            train_size=SplitSize.TRAIN.value,
            random_state=2502,
        )

        val_files, test_files = train_test_split(
            val_test_files,
            test_size=SplitSize.TEST.value
            / (SplitSize.TEST.value + SplitSize.VAL.value),
            random_state=2502,
        )

        all_classes_split[key] = [train_files, val_files, test_files]

    return all_classes_split


def checkImageShapes(samples: dict[str, list[str]]) -> list:
    image_shapes_dict = {}
    in_class_shape_consistency = {}
    out_of_class_shape_sample = []

    for key, value in samples.items():
        image_shapes_dict[key] = [np.shape(cv2.imread(element)) for element in value]

        in_class_shape_consistency[key] = all(
            element == image_shapes_dict[key][0] for element in image_shapes_dict[key]
        )

        out_of_class_shape_sample.append(image_shapes_dict[key][0])

    out_of_class_shape_consistency = all(
        element == out_of_class_shape_sample[0] for element in out_of_class_shape_sample
    )

    return [
        image_shapes_dict,
        in_class_shape_consistency,
        out_of_class_shape_consistency,
    ]


def preProcess(split_sample: dict, encoding: dict) -> list:
    train = []
    label_train = []
    for key, value in split_sample.items():
        for element in value[0]:
            train.append(cv2.imread(element))
            label_train.append(key)

    # One hot encode labels
    for key, value in encoding.items():
        for i in range(len(label_train)):
            if label_train[i] == key:
                label_train[i] = value

    aux = list(zip(train, label_train))
    aux = random.sample(aux, len(aux))
    train, label_train = zip(*aux)

    val = []
    label_val = []
    for key, value in split_sample.items():
        for element in value[0]:
            val.append(cv2.imread(element))
            label_val.append(key)

    # One hot encode labels
    for key, value in encoding.items():
        for i in range(len(label_val)):
            if label_val[i] == key:
                label_val[i] = value

    aux = list(zip(val, label_val))
    aux = random.sample(aux, len(aux))
    val, label_val = zip(*aux)

    test = []
    label_test = []
    for key, value in split_sample.items():
        for element in value[0]:
            test.append(cv2.imread(element))
            label_test.append(key)

    # One hot encode labels
    for key, value in encoding.items():
        for i in range(len(label_test)):
            if label_test[i] == key:
                label_test[i] = value

    aux = list(zip(test, label_test))
    aux = random.sample(aux, len(aux))
    test, label_test = zip(*aux)

    image_data_generator = (
        tf.keras.preprocessing.image.ImageDataGenerator()
    )  # chose not to preprocess yet

    train_generator = image_data_generator.flow(
        x=np.array(train),
        y=np.array(label_train),
        batch_size=20,
    )

    val_generator = image_data_generator.flow(
        x=np.array(val),
        y=np.array(label_val),
        batch_size=5,
    )

    test_generator = image_data_generator.flow(
        x=np.array(test),
        y=np.array(label_test),
        batch_size=5,
    )

    return [train_generator, val_generator, test_generator]


def trainModel(train_generator, val_generator):
    # Define model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(
                    227,
                    227,
                    3,
                ),  # could not use enum because it has no __index__ attribute
            ),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                128, activation="relu"
            ),  # vary into other architectures
            tf.keras.layers.Dense(5, activation="softmax"),
        ]
    )

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train model
    model.fit(train_generator, validation_data=val_generator, epochs=50)

    return model


def evaluateModel(model, generator_list):
    test_loss, test_accuracy = model.evaluate(generator_list[2])

    return [test_loss, test_accuracy]


if __name__ == "__main__":
    class_file_paths = {
        Classes.BROKEN.value: r"/Users/sandrolobao/Desktop/Projects/Personal Projects/Computer Vision/Soybean Seeds/Broken soybeans/",
        Classes.IMMATURE.value: r"/Users/sandrolobao/Desktop/Projects/Personal Projects/Computer Vision/Soybean Seeds/Immature soybeans",
        Classes.INTACT.value: r"/Users/sandrolobao/Desktop/Projects/Personal Projects/Computer Vision/Soybean Seeds/Intact soybeans",
        Classes.SKIN_DAMAGED.value: r"/Users/sandrolobao/Desktop/Projects/Personal Projects/Computer Vision/Soybean Seeds/Skin-damaged soybeans",
        Classes.SPOTTED.value: r"/Users/sandrolobao/Desktop/Projects/Personal Projects/Computer Vision/Soybean Seeds/Spotted soybeans",
    }

    encoding = {
        Classes.BROKEN.value: [1, 0, 0, 0, 0],
        Classes.IMMATURE.value: [0, 1, 0, 0, 0],
        Classes.INTACT.value: [0, 0, 1, 0, 0],
        Classes.SKIN_DAMAGED.value: [0, 0, 0, 1, 0],
        Classes.SPOTTED.value: [0, 0, 0, 0, 1],
    }

    samples = loadImageFiles(class_file_paths)
    split_classes = splitDatasets(samples)
    (
        image_shapes,
        in_class_shape_consistency,
        out_of_class_shape_consistency,
    ) = checkImageShapes(samples)

    generator_list = preProcess(split_sample=split_classes, encoding=encoding)

    model = trainModel(generator_list[0], generator_list[1])

    # test_loss, test_accuracy = evaluateModel(model, generator_list)
