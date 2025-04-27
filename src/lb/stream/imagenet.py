import os
from time import time

import lightning as L
import torch
import torchvision.transforms.v2 as T
import typer
from litdata import StreamingDataLoader, StreamingDataset
from tqdm import tqdm

from lb.utils import clear_cache_dir, default_cache_dir, to_rgb


def stream_imagenet(
    input_dir: str = typer.Option(..., help="Path to the dataset directory"),
    cache_dir: str = typer.Option(
        default_cache_dir(), help="Path to the cache directory"
    ),
    dtype: str = typer.Option("float32", help="Data type: float32 or float16"),
    batch_size: int = typer.Option(256, help="Batch size for benchmarking"),
    num_workers: int = typer.Option(os.cpu_count(), help="Number of workers for dataloader"),
    epochs: int = typer.Option(2, help="Number of epochs to run benchmark"),
    max_cache_size: str = typer.Option(
        "200GB", help="Max cache size for streaming dataset"
    ),
    clear_cache: bool = typer.Option(
        True,
        help="Clear the cache directory before and after running the benchmark",
        show_default=True,
    ),
):
    """
    Benchmark ImageNet streaming dataset.
    """
    L.seed_everything(42)

    if clear_cache:
        clear_cache_dir(cache_dir)

    class ImageNetStreamingDataset(StreamingDataset):
        def __init__(self, *args, **kwargs):
            self.transform = T.Compose(
                [
                    T.RandomResizedCrop(224, antialias=True),
                    T.RandomHorizontalFlip(),
                    T.ToDtype(
                        torch.float32 if dtype == "float32" else torch.float16,
                        scale=True,
                    ),
                ]
            )
            super().__init__(*args, **kwargs)

        def __getitem__(self, index):
            img, class_index = super().__getitem__(index)
            return self.transform(to_rgb(img)), int(class_index)

    dataloader = StreamingDataLoader(
        ImageNetStreamingDataset(
            input_dir=input_dir,
            cache_dir=cache_dir,
            max_cache_size=max_cache_size,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    for epoch in range(epochs):
        num_samples = 0
        t0 = time()
        for data in tqdm(dataloader, smoothing=0, mininterval=1):
            num_samples += data[0].squeeze(0).shape[0]
        print(
            f"For benchmark on {epoch}, streamed over {num_samples} samples in {time() - t0} or {num_samples / (time() - t0)} images/sec."
        )

    clear_cache_dir(cache_dir)
    print("Finished benchmarking.")
