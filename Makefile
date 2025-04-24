.PHONY: benchmark

benchmark:
	@pip install -U -r requirements.txt
	@chmod +x ./scripts/benchmark.sh
	@sh ./scripts/benchmark.sh
