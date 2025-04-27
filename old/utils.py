import json
import os
import shutil
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch


@lru_cache(maxsize=1)
def load_imagenet_class_index():
    """
    Load the ImageNet class index mapping from class names to class indices.

    Returns:
        dict: Mapping from class names to their corresponding index.

    Raises:
        RuntimeError: If the class index file cannot be fetched or parsed.
    """
    # URL for the class index mapping file
    class_index_url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"

    try:
        # Use requests to fetch the file content
        response = requests.get(class_index_url, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the JSON data
        class_index_data = response.json()

        # Create mapping from class name to index
        return {v[0]: int(k) for k, v in class_index_data.items()}

    except (requests.RequestException, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load ImageNet class index: {e}")


class_names_to_index_map = load_imagenet_class_index()


def get_class_index_from_filepath(filepath):
    class_name = filepath.split("/")[-2]
    return int(class_names_to_index_map[class_name])


def shuffle(x):
    return np.random.permutation(x).tolist()


def to_rgb(img):
    if img.shape[0] == 1:
        img = img.repeat((3, 1, 1))
    if img.shape[0] == 4:
        img = img[:3]
    return img


@lru_cache(maxsize=1)
def load_imagenet_val_class_names():
    """
    Load ImageNet validation class names from the official source.

    Returns:
        List[str]: List of ImageNet validation class names (synsets).

    Raises:
        RuntimeError: If the validation labels file cannot be fetched or read.
    """
    # URL for the validation labels file
    val_labels_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt"

    try:
        # Use requests to fetch the file content directly
        response = requests.get(val_labels_url, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Split the content into class names
        return response.text.strip().split("\n")

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to load ImageNet validation class names: {e}")


def check_extensions(filenames, extensions) -> None:
    """Validate filename extensions.

    Args:
        filenames (List[str]): List of files.
        extensions (Set[str]): Acceptable extensions.
    """
    for f in filenames:
        idx = f.rindex(".")
        ext = f[idx + 1 :]
        assert ext.lower() in extensions


def get_classes(
    filenames: List[str], class_names: Optional[List[str]] = None
) -> Tuple[List[int], List[str]]:
    """Get the classes for a dataset split of sample image filenames.

    Args:
        filenames (List[str]): Files, in the format ``"root/split/class/sample.jpeg"``.
        class_names (List[str], optional): List of class names from the other splits that we must
            match. Defaults to ``None``.

    Returns:
        Tuple[List[int], List[str]]: Class ID per sample, and the list of unique class names.
    """
    classes = []
    dirname2class = {}
    for f in filenames:
        d = f.split(os.path.sep)[-2]
        c = dirname2class.get(d)
        if c is None:
            c = len(dirname2class)
            dirname2class[d] = c
        classes.append(c)
    new_class_names = sorted(dirname2class)
    if class_names is not None:
        assert class_names == new_class_names
    return classes, new_class_names


def clear_cache(cache_dir: str) -> None:
    """Clear the cache directory."""
    try:
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
    except Exception as e:
        # print(f"Error clearing cache: {e}")
        pass


def to_rgb(img):
    if isinstance(img, torch.Tensor):
        if img.shape[0] == 1:
            img = img.repeat((3, 1, 1))
        if img.shape[0] == 4:
            img = img[:3]
    else:
        if img.mode == "L":
            img = img.convert("RGB")
    return img
