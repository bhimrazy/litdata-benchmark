import io
import os
from functools import partial
from time import time
from typing import Literal, Optional, Tuple, Union

import numpy as np
import typer
from lightning import seed_everything
from litdata import optimize, walk
from PIL import Image
from tqdm import tqdm

from lb.utils import (
    class_names_to_index_map,
    load_imagenet_class_index,
    load_imagenet_val_class_names,
)


def optimize_imagenet(
    input_dir: str = typer.Option(help="Input directory for raw dataset"),
    output_dir: str = typer.Option(help="Output directory for optimized dataset"),
    write_mode: str = typer.Option(
        None, help="Store images in specified format. ie.e jpeg, pil (raw)"
    ),
    quality: Optional[int] = typer.Option(
        None, help="JPEG quality if JPEG is selected"
    ),
    resize: bool = typer.Option(False, help="Resize images"),
    resize_size: int = typer.Option(None, help="Size to resize images"),
    num_workers: int = typer.Option(8, help="Number of workers for optimization"),
    chunk_bytes: str = typer.Option("64MB", help="Chunk size for optimization"),
):
    """
    Optimize ImageNet dataset for benchmarking.
    """
    seed_everything(42)

    def get_class_from_filepath(filepath: str, classes):
        class_name = filepath.split("/")[-2]
        return classes[class_name]

    def get_inputs(input_dir: str):
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

        # convert to rgb
        if img.mode != "RGB":
            img = img.convert("RGB")

        # resize
        if resize and resize_size is not None:
            if isinstance(resize_size, int):
                # resize the max dimension to resize_size
                max_dim = max(img.size)
                scale = resize_size / max_dim
                new_size = tuple(int(dim * scale) for dim in img.size)
                img = img.resize(new_size)
            elif isinstance(resize_size, tuple) and len(resize_size) == 2:
                img = img.resize(resize_size)

        # write mode
        if args.get("write_mode") == "jpeg":
            buff = io.BytesIO()
            img.save(buff, format="JPEG", quality=args["quality"])
            buff.seek(0)
            img = Image.open(buff)
        elif args.get("write_mode") == "pil":
            img = Image.frombytes(img.mode, img.size, img.tobytes())
        return img, class_index

    args = dict(
        resize=resize,
        resize_size=resize_size,
        write_mode=write_mode,
        quality=quality,
    )
    inputs = get_inputs(input_dir)

    start_time = time()
    optimize(
        fn=partial(optimize_fn, args=args),
        inputs=inputs,
        output_dir=output_dir,
        chunk_bytes=chunk_bytes,
        reorder_files=False,
        num_downloaders=10,
        num_workers=num_workers,
    )
    end_time = time()
    print(f"Time taken to optimize dataset: {end_time - start_time} seconds")
    print("Done!")
