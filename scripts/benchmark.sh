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
echo "Running benchmark and optimization scripts..."

python src/set_optimized_dataset_name.py

FILE="optimized_dataset_name.txt"

if [ ! -e "$FILE" ]; then
  echo "Couldn't write to $FILE. Please check the script.">"$RESULT_FILE"
  exit 1
elif [ ! -s "$FILE" ]; then
  echo "File is empty. Please check the script." > "$RESULT_FILE"
  exit 1
fi

# --- optimize & stream datset ---

RESULT_FILE="result.md"

run_and_check() {
    CMD="$1"
    echo "Running: $CMD"
    eval "$CMD"
    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "### ❌ Error: Command failed — $CMD (exit code $STATUS)"
        # Create a temp file with the error message + old content
        {
            echo -e "❌ **Error occurred while running:** \`$CMD\` (exit code $STATUS)\n"
            cat "$RESULT_FILE"
        } > "${RESULT_FILE}.tmp"
        mv "${RESULT_FILE}.tmp" "$RESULT_FILE"
        exit $STATUS
    fi
}

run_and_check "python src/optimize/optimize_imagenet.py"
run_and_check "python src/stream/stream_imagenet.py"
