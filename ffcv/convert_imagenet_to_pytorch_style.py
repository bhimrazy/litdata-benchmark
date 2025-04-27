import json
import os
from argparse import ArgumentParser
from functools import lru_cache
from glob import glob

import requests
from torchvision.datasets import ImageFolder
from tqdm import tqdm


@lru_cache(maxsize=1)
def load_imagenet_class_index():
    """
    Load the ImageNet class index mapping from class names to class indices.

    Returns:
        dict: Mapping from class names to their corresponding index.

    Raises:
        RuntimeError: If the class index file cannot be fetched or parsed.
    """
    # URL for the class index mapping file
    class_index_url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"

    try:
        # Use requests to fetch the file content
        response = requests.get(class_index_url, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the JSON data
        class_index_data = response.json()

        # Create mapping from class name to index
        return {v[0]: int(k) for k, v in class_index_data.items()}

    except (requests.RequestException, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load ImageNet class index: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert Imagenet Dataset to PyTorch style inplace."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the ImageNet dataset directory.",
    )

    args = parser.parse_args()
    imagenet_dir = args.data_dir
    # Check if the directory exists
    if not os.path.exists(imagenet_dir):
        raise FileNotFoundError(
            f"The specified directory does not exist: {imagenet_dir}"
        )

    # Load the ImageNet class index mapping
    class_index_mapping = load_imagenet_class_index()

    # create all the folders
    for _, class_index in class_index_mapping.items():
        folder_path = f"{imagenet_dir}/{class_index}"
        os.makedirs(folder_path, exist_ok=True)

    # Move the files to their respective folders
    for file_path in tqdm(glob(f"{imagenet_dir}/*/**.*"), desc="Moving files"):
        # Filter files in folders starting with 'n'
        dirname = os.path.basename(os.path.dirname(file_path))
        if not dirname.startswith("n"):
            continue
        class_index = class_index_mapping[dirname]
        destination_path = f"{imagenet_dir}/{class_index}/{os.path.basename(file_path)}"
        os.rename(file_path, destination_path)

    # Remove the old folders
    for folder in tqdm(glob(f"{imagenet_dir}/*"), desc="Removing old folders"):
        # Remove folders starting with 'n'
        if  os.path.basename(folder).startswith("n"):
            os.rmdir(folder)
    print("Conversion complete.")
    print("All folders have been moved to their respective class index folders.")
    print("Old folders have been removed.")
    print("You can now use the dataset in PyTorch style.")

    dataset = ImageFolder(root=imagenet_dir, transform=None)
    classes = dataset.classes
    print("Classes in the dataset:", classes)
    print("Number of classes:", len(classes))
    print("sample dataset:", dataset[0])
