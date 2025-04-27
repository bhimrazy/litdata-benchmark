import os
from pathlib import Path
from time import time

from datasets import load_dataset
from lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import clear_cache

SHUFFLE: bool = bool(int(os.getenv("SHUFFLE", 0)))
DATASET: bool = bool(int(os.getenv("DATASET", 0)))


if __name__ == "__main__":
    # Fixed the seed across packages
    SEED = 42
    seed_everything(SEED)

    # Clean cache
    cache_dir = str(Path.home() / ".cache" / "hf-datasets")
    clear_cache(cache_dir)

    DATASETS = [
        {
            "path": "open-thoughts/OpenThoughts-114k",
            "name": "default",
            "split": "train",
        },
        {
            "path": "HuggingFaceFW/fineweb-edu",
            "name": "sample-10BT",
            "split": "train",
        },
    ]
    ACTIVE_DATASET = DATASETS[DATASET]
    path = ACTIVE_DATASET["path"]
    name = ACTIVE_DATASET["name"]
    split = ACTIVE_DATASET["split"]
    print(f"Shuffle: {SHUFFLE} Dataset: {path}/{name}")

    dataset = load_dataset(
        path=path, name=name, split=split, cache_dir=cache_dir, streaming=True
    )

    seed, buffer_size = SEED, 10_000
    if SHUFFLE:
        dataset = dataset.shuffle(seed, buffer_size=buffer_size)

    # Define the DataLoader
    dataloader = DataLoader(
        dataset, batch_size=256, num_workers=os.cpu_count(), drop_last=True
    )  # doesn't seem to accept num_workers > num_shards

    # Iterate over the datasets for 2 epochs
    for epoch in range(2):
        num_samples = 0
        t0 = time()
        dataset.set_epoch(epoch)
        for i, data in enumerate(tqdm(dataloader, smoothing=0, mininterval=1)):
            num_samples += len(data[list(data.keys())[0]])
        print(
            f"For {__file__} on {epoch}, streamed over {num_samples} samples in {time() - t0} or {num_samples / (time() - t0)} samples/sec."
        )

    # Cleanup cache
    clear_cache(cache_dir)
    print("Finished benchmarking.")
