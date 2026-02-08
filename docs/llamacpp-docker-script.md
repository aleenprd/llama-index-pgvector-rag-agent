# Llama.cpp Docker Server Script

## Overview

`start-docker-llamacpp-server.sh` is an interactive script that runs a llama.cpp inference server in Docker. It provides a user-friendly interface for selecting and running local Large Language Models (LLMs) in GGUF format with GPU or CPU acceleration.

## Purpose

This script simplifies the process of running local LLMs by:
- **Model Discovery**: Recursively finds all GGUF models in your models directory
- **Smart Selection**: Displays models sorted by size (largest first) for easy selection
- **GPU Support**: Automatically detects GPU availability and falls back to CPU if needed
- **Flexible Configuration**: Supports environment variables and command-line arguments
- **Container Management**: Handles Docker lifecycle and provides useful management commands

## Basic Usage

### Quick Start

Run the script and follow the interactive prompts:

```bash
./start-docker-llamacpp-server.sh
```

The script will:
1. Scan for GGUF models in your models directory
2. Display them sorted by size with their paths
3. Let you select a model
4. Check for GPU availability
5. Start the llama.cpp server in Docker

### Model Directory

By default, the script looks for models in the path specified by `LLAMACPP_MODELS_PATH`. Make sure to:
- Set this environment variable
- Or pass the path via command-line argument
- Or provide it when prompted

### Accessing the Server

Once started, the server will be available at:
```
http://localhost:8000
```

(or the port you configured via `LLAMACPP_PORT`)

## Configuration

### Environment Variables

The script recognizes these environment variables:

- `LLAMACPP_HOST` - Server host (default: 0.0.0.0)
- `LLAMACPP_PORT` - Server port (default: 8000)
- `LLAMACPP_MODELS_PATH` - Path to GGUF models directory (required)
- `LLAMACPP_CONTEXT_SIZE` - Context window size (default: 512)
- `LLAMACPP_GPU_LAYERS` - GPU layers to offload (default: 99)
- `LLAMACPP_LOG_FILE` - Log file name (default: llama-server.log)
- `LLAMA_CPP_IMAGE` - Docker image (default: ghcr.io/ggml-org/llama.cpp:full-cuda)
- `LLAMACPP_CUDA_TEST_IMAGE` - CUDA test image for GPU detection (default: nvidia/cuda:12.6.0-base-ubuntu24.04)

### Command-Line Arguments

View all available options:

```bash
./start-docker-llamacpp-server.sh --help
```

Override settings with arguments:
```bash
./start-docker-llamacpp-server.sh --port 9000 --context-size 2048 --gpu-layers 50
```

## Features

### Recursive Model Discovery

The script recursively searches your models directory for all `.gguf` files, regardless of subdirectory structure. This means you can organize models however you like:

```
models/
├── qwen/
│   └── model.gguf
├── llama/
│   ├── 7b/
│   │   └── model.gguf
│   └── 13b/
│       └── model.gguf
└── mixtral.gguf
```

All models will be found and presented for selection.

### Size-Based Sorting

Models are displayed sorted by file size (largest first), making it easy to:
- Identify the most capable models (typically larger)
- See model size at a glance
- Make informed choices based on available resources

### GPU Detection

The script automatically:
- Tests for NVIDIA GPU availability via NVIDIA Container Toolkit
- Prompts to use CPU mode if GPU is not available
- Falls back gracefully without manual intervention

See [`docs/gpu-setup.md`](./gpu-setup.md) for GPU setup instructions.

### Run Modes

Choose between two execution modes:

1. **Foreground Mode**: 
   - Server runs in terminal
   - See live logs
   - Stop with Ctrl+C

2. **Detached Mode**: 
   - Server runs in background
   - Frees up terminal
   - View logs with `docker logs llama-cpp-server`

## Container Details

- **Container Name**: `llama-cpp-server`
- **Default Image**: `ghcr.io/ggml-org/llama.cpp:full-cuda`
- **Exposed Port**: Configurable (default: 8000)
- **GPU Support**: Requires NVIDIA Container Toolkit
- **Models Mount**: Your models directory is mounted as `/models` in container

## Model Format

The script works with GGUF format models, which are:
- Quantized for efficient inference
- Self-contained with all weights and config
- Available from sources like:
  - Hugging Face (search for "GGUF")
  - LM Studio model library
  - Manual conversions with llama.cpp tools

## Common Use Cases

### Local Development

Run a small model for testing:
```bash
LLAMACPP_MODELS_PATH=/home/user/models ./start-docker-llamacpp-server.sh
```

Select a 4B or 7B parameter model for quick responses.

### Production Inference

Run with larger context and specific settings:
```bash
./start-docker-llamacpp-server.sh \
  --models-path /data/llm-models \
  --context-size 4096 \
  --port 8080 \
  --gpu-layers 99
```

### CPU-Only Environment

The script will prompt to use CPU mode if GPU is unavailable:
```
No GPU detected or NVIDIA Container Toolkit not installed.
Do you want to run the server in CPU-only mode? (y/n):
```

## Managing the Server

### View Logs
```bash
docker logs llama-cpp-server
```

### Follow Logs (Live)
```bash
docker logs -f llama-cpp-server
```

### Stop Server
```bash
docker stop llama-cpp-server
```

### Remove Container
```bash
docker rm llama-cpp-server
```

### Restart with Different Model
Stop and remove the existing container, then run the script again.

## API Integration

Once running, use the server with OpenAI-compatible clients:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local-model",  # Model name doesn't matter for llama.cpp
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Troubleshooting

### No GPU Detected

Install NVIDIA Container Toolkit (see [`docs/gpu-setup.md`](./gpu-setup.md)) or run in CPU mode.

### Port Already in Use

Specify a different port:
```bash
LLAMACPP_PORT=8001 ./start-docker-llamacpp-server.sh
```

### No Models Found

Ensure:
- `LLAMACPP_MODELS_PATH` points to the correct directory
- Directory contains `.gguf` files
- You have read permissions

### Container Exists

The script will prompt to remove existing containers. Choose "yes" to replace with new configuration.

### Out of Memory

Reduce context size or GPU layers:
```bash
./start-docker-llamacpp-server.sh --context-size 512 --gpu-layers 30
```

## Performance Tuning

### GPU Layers (`-ngl`)
- Higher values = more GPU offloading = faster inference
- Set to 99 to offload all layers
- Reduce if running out of VRAM

### Context Size (`-c`)
- Larger context = more memory usage
- Common sizes: 512, 1024, 2048, 4096, 8192
- Depends on model and available RAM/VRAM

### Host Binding
- Default `0.0.0.0` allows external connections
- Use `127.0.0.1` to restrict to localhost only

## Future Enhancements

The script may be extended to support:
- Multiple concurrent model servers
- Model variant selection (Q4, Q5, Q8 quantizations)
- Advanced llama.cpp parameters (temperature, top-k, etc.)
- LoRA adapter support
- Multi-GPU configurations
- Benchmarking and performance metrics
- Model preloading and warm-up
- Load balancing across instances
- Custom system prompts and templates
- Integration with LangChain, LlamaIndex, etc.

## Related Documentation

- [GPU Setup Guide](./gpu-setup.md)
- [Makefile Usage](./makefile-usage.md)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [GGUF Format Specification](https://github.com/ggml-org/gguf)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
