import uuid

import litdata as ld  # make sure litdata is installed

def main():
    # Generate a random UUID
    random_uuid = str(uuid.uuid4())

    # Get LitData version
    version = ld.__version__

    # Construct the directory name
    optimized_dataset_dir = f"{version}__{random_uuid}"

    # write to file
    with open("optimized_dataset_dir.txt", "w") as f:
        f.write(optimized_dataset_dir)
    print(f"Successfully set environment variable: OPTIMIZED_DATASET_DIR={optimized_dataset_dir}")

if __name__ == "__main__":
    main()
