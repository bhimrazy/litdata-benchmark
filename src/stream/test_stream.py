from stream_imagenet import ImageNetStreamingDataset, StreamingDataLoader

if __name__ == "__main__":
    dataset = ImageNetStreamingDataset(
        input_dir="/teamspace/datasets/imagenet-1m-optimized-0.2.41/train",
        cache_dir="cache",
        max_cache_size="200GB",
    )

    print("Streaming dataset created.", len(dataset))
    print("Sample data:", dataset[100000])

    dataloader = StreamingDataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
    )

    batch = next(iter(dataloader))
    print("Batch data:", batch)
