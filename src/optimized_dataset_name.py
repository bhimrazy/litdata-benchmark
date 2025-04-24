import uuid

import litdata as ld  # make sure litdata is installed

def get_optimized_dataset_name() -> str:
    with open("optimized_dataset_name.txt", "r") as f:
        optimized_dataset_name = f.read().strip()
    return optimized_dataset_name

def main():
    # Generate a random UUID
    random_uuid = str(uuid.uuid4())

    # Get LitData version
    version = ld.__version__

    # Construct the directory name
    optimized_dataset_name = f"{version}__{random_uuid}"

    # write to file
    with open("optimized_dataset_name.txt", "w") as f:
        f.write(optimized_dataset_name)
    print(f"Successfully set : OPTIMIZED_DATASET_NAME={optimized_dataset_name}")

if __name__ == "__main__":
    main()

