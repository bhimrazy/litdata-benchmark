import os
from time import time

import lightning as L
import torch
import torchvision.transforms.v2 as T
from litdata import StreamingDataLoader, StreamingDataset, __version__
from tqdm import tqdm
from utils import clear_cache, to_rgb


# Create a custom streaming dataset for Imagenet
class ImageNetStreamingDataset(StreamingDataset):
    def __init__(self, *args, **kwargs):
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(224, antialias=True),
                T.RandomHorizontalFlip(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Note: If torchvision is installed, we return a tensor image instead of a pil image as it is much faster.
        img, class_index = super().__getitem__(
            index
        )  # <- Whatever you returned from the DatasetOptimizer prepare_item method.
        return self.transform(to_rgb(img)), int(class_index)
        # return self.transform(
        #     to_rgb(img)
        # ), class_index  # int cannot be used as class_index is a filepath string


if __name__ == "__main__":
    # Fixed the seed across packages
    L.seed_everything(42)

    print(f"Benchmarking using litdata version: {__version__}")

    # Clean cache
    cache_dir = "/cache/chunks/"
    clear_cache(cache_dir)
    
    # # Uncomment the following lines to use DDP
    # fabric = L.Fabric(strategy="ddp", accelerator="cpu", devices=2)
    # fabric.launch()

    # Define the DataLoader
    dataloader = StreamingDataLoader(
        ImageNetStreamingDataset(
            input_dir="/teamspace/datasets/imagenet-1m-optimized-0.2.41-v2/train",
            max_cache_size="200GB",
        ),
        batch_size=256,
        num_workers=os.cpu_count(),  # type: ignore
        # profile_batches=10,
    )

    # Iterate over the datasets for 2 epochs
    for epoch in range(2):
        num_samples = 0
        t0 = time()
        for data in tqdm(dataloader, smoothing=0, mininterval=1):
            num_samples += data[0].squeeze(0).shape[0]
            # torch.distributed.barrier() # Uncomment to use DDP
        print(
            f"For {__file__} on {epoch}, streamed over {num_samples} samples in {time() - t0} or {num_samples / (time() - t0)} images/sec."
        )

    # Cleanup cache
    clear_cache(cache_dir)
    print("Finished benchmarking.")
