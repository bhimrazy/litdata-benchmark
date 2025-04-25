import os
import sys
from time import time
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Any

import numpy as np
from lightning import seed_everything
from litdata import optimize, walk
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    class_names_to_index_map,
    clear_cache,
    load_imagenet_val_class_names,
    load_imagenet_class_index,
)
from optimized_dataset_name import get_optimized_dataset_name

RESULT_FILE = "result.md"

def parse_args() -> Namespace:
    """Parse command-line arguments.

    Args:
        Namespace: command-line arguments.
    """
    args = ArgumentParser()

    optimized_dir = get_optimized_dataset_name()

    args.add_argument(
        "--input_dir",
        default="/teamspace/s3_connections/imagenet-1m-template/raw/train",
        type=str,
        help="Input directory where the data lives",
    )
    args.add_argument(
        "--output_dir",
        default=f"/teamspace/datasets/imagenet-1m-optimized/{optimized_dir}",
        type=str,
        help="Output directory where the optimized dataset will be stored",
    )
    args.add_argument(
        "--resize",
        action="store_true",
        help="Whether to resize the images to (224, 244).",
    )
    args.add_argument(
        "--filepath_only",
        action="store_true",
        help="Whether to store only the image and the original filepath.",
    )
    args.add_argument(
        "--use_jpeg_90",
        action="store_true",
        help="Whether to store the images as JPEG.",
    )
    return args.parse_args()


def get_class_from_filepath(filepath: str, classes) -> int:
    class_name = filepath.split("/")[-2]
    return classes[class_name]


def get_inputs(input_dir: str) -> Any:
    classes = load_imagenet_class_index()
    filepaths = np.random.permutation(
        [
            os.path.join(root, filename)
            for root, _, filenames in tqdm(walk(input_dir), smoothing=0)
            for filename in filenames
        ]
    )
    if "train" in input_dir:
        return [
            (filepath, get_class_from_filepath(filepath, classes))
            for filepath in filepaths
        ]

    class_names = load_imagenet_val_class_names()
    return [
        (filepath, class_names_to_index_map[class_name])
        for filepath, class_name in zip(filepaths, class_names)
    ]


def optimize_fn(data, args):
    filepath, class_index = data
    img = Image.open(filepath)
    # print(f"Processing file: {filepath}, Image shape: {img.size}")
    # if args.resize:
    #     img = img.resize((224, 224))

    # if not args.filepath_only:
    #     return img, class_index

    # # Used only to build optimized search indexes
    # filepath = "/".join(filepath.split("/")[3:])

    # if args.use_jpeg_90:
    #     buff = io.BytesIO()
    #     img.convert("RGB").save(buff, format="JPEG", quality=90)
    #     buff.seek(0)
    #     img = Image.open(buff)

    return img, class_index

def write_to_file(filepath: str, content: str)->None:
    with open(filepath, "w+") as f:
        f.write(content)
    print(f"Written to {filepath}")

if __name__ == "__main__":
    seed_everything(42)
    cache_dir = "/cache/data"
    clear_cache(cache_dir)

    args = parse_args()
    print(f"Arguments: {vars(args)}")
    

    inputs = get_inputs(args.input_dir)


    dataset_len = len(inputs)
    write_to_file(RESULT_FILE, f"Dataset length: {dataset_len}")
    start_time = time()
    optimize(
        fn=partial(optimize_fn, args=args),
        inputs=inputs,
        output_dir=args.output_dir,
        chunk_bytes="64MB",
        reorder_files=False,
        num_downloaders=10,
        num_workers=16,
        mode="overwrite",
    )
    end_time = time()
    write_to_file(RESULT_FILE, f"Time taken to optimize dataset: {end_time - start_time} seconds")
    clear_cache(cache_dir)
    print("Done!")
