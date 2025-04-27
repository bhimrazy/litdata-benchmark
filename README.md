# LitData Benchmark CLI

A modern, modular CLI for benchmarking and optimizing datasets (e.g., ImageNet) with [LitData](https://lightning.ai/pages/litdata/).

---

## 🚀 Features
- **Stream & Benchmark**: Fast, reproducible streaming benchmarks for large datasets.
- **Optimize**: Prepare and optimize datasets for efficient streaming.
- **Extensible**: Add new datasets and commands with ease.
- **Cloud Ready**: (Coming soon) Run on Lightning AI cloud with a single flag.

---

## 🛠️ Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/bhimrazy/litdata-benchmark
    cd litdata-benchmark
    ```
2. **Install dependencies:**
    ```sh
    pip install -U -r requirements.txt
    ```
    Or, for editable install with CLI:
    ```sh
    pip install -e .
    ```

---

## 🏁 Quickstart

### 1. **Optimize ImageNet Dataset**
Prepare your dataset for streaming:
```sh
lb optimize imagenet --input-dir /path/to/raw --output-dir /path/to/optimized --resize --jpeg
```

### 2. **Stream & Benchmark**
Run a streaming benchmark:
```sh
lb stream imagenet --input-dir /path/to/optimized --batch-size 256 --dtype float32
```

See all options:
```sh
lb --help
lb optimize imagenet --help
lb stream imagenet --help
```

---

## 📂 Project Structure

```
src/lb/           # Main CLI and logic
    cli.py        # CLI entrypoint
    optimize/     # Optimization commands (e.g., imagenet)
    stream/       # Streaming/benchmark commands (e.g., imagenet)
    utils.py      # Shared utilities
scripts/          # Helper scripts (e.g., for HF datasets, cloud)
tests/            # Test suite
```

---

## 🧑‍💻 Contributing
- Open issues or pull requests for bugs, features, or improvements.
- Please lint and test your code before submitting.

---

## 📄 License
MIT License

---

## 🙏 Acknowledgements
- Built on [LitData](https://lightning.ai/pages/litdata/) and [Lightning AI](https://lightning.ai/).
- Inspired by the [LitData Benchmarking Guide](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries?view=org&section=featured).

---

Happy benchmarking! 🚦
