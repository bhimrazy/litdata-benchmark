.PHONY: benchmark

benchmark:
	# @pip install -U -r requirements.txt
	# @chmod +x ./scripts/benchmark.sh
	# @sh ./scripts/benchmark.sh
	@pip install -U -e.
	@lb stream imagenet --input-dir /teamspace/datasets/imagenet-1m-optimized-0.2.41-v2/train/


