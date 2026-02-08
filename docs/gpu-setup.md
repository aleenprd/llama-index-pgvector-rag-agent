# GPU Setup for Docker (NVIDIA Container Toolkit)

## Overview

To use NVIDIA GPUs with Docker containers (like llama.cpp server), you need to install the NVIDIA Container Toolkit. This enables Docker to access your GPU hardware.

## Prerequisites

- NVIDIA GPU with compatible drivers installed
- Docker installed and running
- Ubuntu/Debian-based Linux distribution (instructions below are for Ubuntu 24.04)

## Installation Steps

### 1. Add NVIDIA's GPG Key

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

### 2. Add the NVIDIA Container Toolkit Repository

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 3. Update Package List

```bash
sudo apt-get update
```

### 4. Install NVIDIA Container Toolkit

```bash
sudo apt-get install -y nvidia-container-toolkit
```

### 5. Configure Docker Runtime

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

### 6. Restart Docker

```bash
sudo systemctl restart docker
```

## Verification

Test that Docker can access your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

You should see output showing your GPU information (model, memory, driver version, etc.).

## Troubleshooting

### Error: "could not select device driver with capabilities: [[gpu]]"

This error means the NVIDIA Container Toolkit is not installed or not configured properly. Follow the installation steps above.

### Check NVIDIA Driver Installation

Verify your NVIDIA drivers are installed on the host system:

```bash
nvidia-smi
```

If this command doesn't work, you need to install NVIDIA drivers first:

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Docker Permission Issues

If you get permission errors, ensure your user is in the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## CPU-Only Fallback

If you don't have a GPU or choose not to install the toolkit, the `start-docker-llamacpp-server.sh` script will prompt you to run in CPU-only mode. Note that CPU inference is significantly slower than GPU inference.

## Additional Resources

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [llama.cpp Documentation](https://github.com/ggml-org/llama.cpp)
