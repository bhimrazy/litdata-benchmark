#!/bin/bash
set -eu

# check if raw imagenet dataset exists or not
# if not, write to result.md file and exit.
# if does, run python scripts to benchmark optimize * streaming.
DIR_PATH="/teamspace/s3_connections/imagenet-1m-template/raw/train"
RESULT_FILE="result.md"

if [ ! -d "$DIR_PATH" ] || [ -z "$(ls -A "$DIR_PATH")" ]; then
  cat <<EOF > "$RESULT_FILE"
No dataset found at $DIR_PATH.

Please make sure raw imagenet dataset is available for optimizing and benchmarking.
EOF
  echo "No dataset found at $DIR_PATH. Please check the result.md file for details."
  exit 1
fi

echo "Raw Imagenet dataset found at $DIR_PATH."
# Proceed with the rest of the script
# python scripts/benchmark.py --dataset_dir "$DIR_PATH"
# python scripts/optimize.py --dataset_dir "$DIR_PATH"
# Add your benchmarking and optimization commands here
echo "Running benchmark and optimization scripts..."

python src/set_env_var.py

FILE="optimized_dataset_dir.txt"

if [ ! -e "$FILE" ]; then
  echo "Couldn't write to $FILE. Please check the script.">"$RESULT_FILE"
  exit 1
elif [ ! -s "$FILE" ]; then
  echo "File is empty. Please check the script." > "$RESULT_FILE"
  exit 1
fi
