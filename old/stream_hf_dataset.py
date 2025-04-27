import os
from time import time

from lightning import seed_everything
from litdata import StreamingDataLoader, StreamingDataset, __version__
from litdata.streaming.item_loader import ParquetLoader
from tqdm import tqdm
from utils import clear_cache
from pathlib import Path

SHUFFLE: bool = bool(int(os.getenv("SHUFFLE", 0)))
PRELOAD: bool = bool(int(os.getenv("PRELOAD", 0)))
LOW_MEMORY = bool(int(os.getenv("LOW_MEMORY", 1)))
DATASET: bool = bool(int(os.getenv("DATASET", 0)))


if __name__ == "__main__":
    # Fixed the seed across packages
    seed_everything(42)
    # Clean cache
    cache_dir = str(Path.home() / ".cache" / "litdata-cache-index-pq")
    clear_cache(cache_dir)

    DATASETS = [
        "hf://datasets/open-thoughts/OpenThoughts-114k/data",
        "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT",
    ]

    ACTIVE_DATASET = DATASETS[DATASET]
    print(f"Benchmarking using litdata version: {__version__}")
    print(
        f"Shuffle: {SHUFFLE}, Preload: {PRELOAD}, Low Memory: {LOW_MEMORY} \nDataset: {ACTIVE_DATASET}"
    )

    # Define the StreamingDataset
    dataset = StreamingDataset(
        input_dir=ACTIVE_DATASET,
        item_loader=ParquetLoader(low_memory=LOW_MEMORY, pre_load_chunk=PRELOAD),
    )

    print("Total number of samples in the dataset:", len(dataset))

    # Define the DataLoader
    dataloader = StreamingDataLoader(
        dataset,
        batch_size=256,
        num_workers=os.cpu_count(),  # type: ignore
        shuffle=SHUFFLE,
    )

    # Iterate over the datasets for 2 epochs
    for epoch in range(2):
        num_samples = 0
        t0 = time()
        for data in tqdm(dataloader, smoothing=0, mininterval=1):
            num_samples += len(data[list(data.keys())[0]])
        print(
            f"For {__file__} on {epoch}, streamed over {num_samples} samples in {time() - t0} or {num_samples / (time() - t0)} samples/sec."
        )

    # Cleanup cache
    clear_cache(cache_dir)
    print("Finished benchmarking.")
