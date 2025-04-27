import os
from time import time

import lightning as L
import torch
import torch.nn as nn
from typing import List
import torchvision.transforms.v2 as T
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    Squeeze,
    NormalizeImage,
    RandomHorizontalFlip,
    ToTorchImage,
    Convert,
    ModuleWrapper,
)
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.fields.basics import IntDecoder
from tqdm import tqdm

# from src.optimized_dataset_name import get_optimized_dataset_name
import numpy as np

RESULT_FILE = "result.md"


def write_to_file(filename: str, content: str) -> None:
    with open(filename, "w") as f:
        f.write(content)
    print(f"Written to {filename}")


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256

if __name__ == "__main__":
    # Fixed the seed across packages
    L.seed_everything(42)

    # Clean cache
    cache_dir = "/cache/chunks/"

    # clear_cache(cache_dir)
    class ScaleBy255(nn.Module):
        def forward(self, x):
            return x / 255.0

    # # Uncomment the following lines to use DDP
    # fabric = L.Fabric(strategy="ddp", accelerator="cpu", devices=2)
    # fabric.launch()

    # Define the DataLoader
    # optimized_dataset_dir_name = get_optimized_dataset_name()
    image_pipeline = [
        RandomResizedCropRGBImageDecoder((224, 224)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToTorchImage(),
        Convert(torch.float32),
        ModuleWrapper(ScaleBy255()),
    ]

    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), Squeeze()]
    dataloader = Loader(
        "data/imagenet-1m-ffcv/train_256_1.0_90.ffcv",
        batch_size=256,
        num_workers=os.cpu_count(),
        # os_cache=True,
        pipelines={"image": image_pipeline, "label": label_pipeline},
    )

    # Iterate over the datasets for 2 epochs
    write_to_file(RESULT_FILE, "\n---\n")
    for epoch in range(2):
        num_samples = 0
        t0 = time()
        for data in tqdm(dataloader, smoothing=0, mininterval=1):
            num_samples += data[0].squeeze(0).shape[0]
            # torch.distributed.barrier() # Uncomment to use DDP
        msg = f"For {__file__} on {epoch}, streamed over {num_samples} samples in {time() - t0} or {num_samples / (time() - t0)} images/sec."
        print(msg)
        write_to_file(RESULT_FILE, msg)

    # Cleanup cache
    # clear_cache(cache_dir)
    print("Finished benchmarking.")
