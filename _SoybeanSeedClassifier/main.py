import os
import cv2
import random

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

        aux_key: str = key
        aux_value: list[str] = [train_files, val_files, test_files]

        all_classes_split[aux_key] = aux_value

    return all_classes_split


def preProcess():
    return


def featureExtraction():
    return


def trainModel():
    return


def evaluateModel():
    return


if __name__ == "__main__":
    class_file_paths = {
        Classes.BROKEN.value: r"/Users/sandrolobao/Desktop/Projects/Personal Projects/Computer Vision/Soybean Seeds/Broken soybeans/",
        Classes.IMMATURE.value: r"/Users/sandrolobao/Desktop/Projects/Personal Projects/Computer Vision/Soybean Seeds/Immature soybeans",
        Classes.INTACT.value: r"/Users/sandrolobao/Desktop/Projects/Personal Projects/Computer Vision/Soybean Seeds/Intact soybeans",
        Classes.SKIN_DAMAGED.value: r"/Users/sandrolobao/Desktop/Projects/Personal Projects/Computer Vision/Soybean Seeds/Skin-damaged soybeans",
        Classes.SPOTTED.value: r"/Users/sandrolobao/Desktop/Projects/Personal Projects/Computer Vision/Soybean Seeds/Spotted soybeans",
    }

    samples = loadImageFiles(class_file_paths)
    split_classes = splitDatasets(samples)
