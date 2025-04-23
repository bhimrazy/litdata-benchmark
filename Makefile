.PHONY: benchmark

benchmark:
	touch result.md
	python stream_imagenet.py > result.md

