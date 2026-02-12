#!/bin/bash

# Llama.cpp Docker Server Launcher
#
# This script can be configured using environment variables and/or command-line arguments.
# Command-line arguments take precedence over environment variables.
#
# Environment variables:
#   LLAMACPP_HOST           - Server host (default: 0.0.0.0)
#   LLAMACPP_PORT           - Server port (default: 8000)
#   LLAMACPP_MODELS_PATH    - Path to model files (default: /models)
#   LLAMACPP_CONTEXT_SIZE   - Context size (default: 512)
#   LLAMACPP_GPU_LAYERS     - GPU layers (default: 99)
#   LLAMACPP_LOG_FILE       - Log file name (default: llama-server.log)
#   LLAMACPP_IMAGE          - Docker image (default: ghcr.io/ggml-org/llama.cpp:full-cuda)
#   LLAMACPP_CUDA_TEST_IMAGE - CUDA test image (default: nvidia/cuda:11.8.0-base-ubuntu22.04)
#
# Command-line arguments (override environment variables):
#   --host               - Server host
#   -p, --port           - Server port
#   -m, --models-path    - Path to model files
#   -c, --context-size   - Context size
#   -g, --gpu-layers     - GPU layers
#   -l, --log-file       - Log file name
#   -i, --image          - Docker image
#   --cuda-test-image    - CUDA test image
#   -h, --help           - Show this help message
#
# Example usage:
#   ./start-docker-llamacpp-server.sh -c 1024 -g 50 -l my-server.log
#   LLAMACPP_PORT=8080 ./start-docker-llamacpp-server.sh --context-size 2048 --port 9000

# Color codes for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}[LLAMA.CPP DOCKER]${NC} $1"
}

print_command() {
    echo -e "${CYAN}[COMMAND]${NC} $1"
}

# Print script header
echo -e "${PURPLE}============================================${NC}"
echo -e "${PURPLE}    Llama.cpp Docker Server Manager        ${NC}"
echo -e "${PURPLE}============================================${NC}"
echo ""

# Check if .env file exists and ask user if they want to source it
if [ -f ".env" ]; then
    print_info "Found .env file in current directory: $(pwd)/.env"
    echo "Do you want to source the .env file to load environment variables? (y/n):"
    read -r source_env
    if [[ "$source_env" =~ ^[Yy]$ ]]; then
        source .env
        print_success "Environment variables loaded from .env file."
    else
        print_warning "Skipping .env file. Using default values."
    fi
    echo ""
fi

# Set default values from environment variables with fallbacks
HOST=${LLAMACPP_HOST:-0.0.0.0}
SERVER_PORT=${LLAMACPP_PORT:-8000}
MODELS_PATH=${LLAMACPP_MODELS_PATH:-/models}
CONTEXT_SIZE=${LLAMACPP_CONTEXT_SIZE:-512}
GPU_LAYERS=${LLAMACPP_GPU_LAYERS:-99}
LOG_FILE=${LLAMACPP_LOG_FILE:-llama-server.log}
LLAMA_CPP_IMAGE=${LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:full-cuda}
CUDA_TEST_IMAGE=${LLAMACPP_CUDA_TEST_IMAGE:-nvidia/cuda:12.6.0-base-ubuntu24.04}

# Parse command-line arguments (these override environment variables)
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -m|--models-path)
            MODELS_PATH="$2"
            shift 2
            ;;
        -c|--context-size)
            CONTEXT_SIZE="$2"
            shift 2
            ;;
        -g|--gpu-layers)
            GPU_LAYERS="$2"
            shift 2
            ;;
        -l|--log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        -i|--image)
            LLAMA_CPP_IMAGE="$2"
            shift 2
            ;;
        --cuda-test-image)
            CUDA_TEST_IMAGE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST               Server host (default: 0.0.0.0)"
            echo "  -p, --port PORT           Server port (default: 8000)"
            echo "  -m, --models-path PATH    Path to model files (default: /models)"
            echo "  -c, --context-size SIZE   Context size (default: 512)"
            echo "  -g, --gpu-layers LAYERS   GPU layers (default: 99)"
            echo "  -l, --log-file FILE       Log file name (default: llama-server.log)"
            echo "  -i, --image IMAGE         Docker image (default: ghcr.io/ggml-org/llama.cpp:full-cuda)"
            echo "  --cuda-test-image IMAGE   CUDA test image (default: nvidia/cuda:12.6.0-base-ubuntu24.04)"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Environment variables (use LLAMACPP_ prefix):"
            echo "  LLAMACPP_HOST              Server host"
            echo "  LLAMACPP_PORT              Server port"
            echo "  LLAMACPP_MODELS_PATH       Path to model files"
            echo "  LLAMACPP_CONTEXT_SIZE      Context size"
            echo "  LLAMACPP_GPU_LAYERS        GPU layers"
            echo "  LLAMACPP_LOG_FILE          Log file name"
            echo "  LLAMACPP_IMAGE             Docker image"
            echo "  LLAMACPP_CUDA_TEST_IMAGE   CUDA test image for GPU detection"
            echo ""
            echo "Examples:"
            echo "  $0 -c 1024 -g 50 -l my-server.log"
            echo "  LLAMACPP_PORT=8080 $0 --context-size 2048 --port 9000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

# Display current configuration
print_header "Current Configuration:"
echo "  🐳 Docker Image: $LLAMA_CPP_IMAGE"
echo "  🌐 Host: $HOST"
echo "  🔌 Port: $SERVER_PORT"
echo "  📁 Models Path: $MODELS_PATH"
echo "  🧠 Context Size: $CONTEXT_SIZE"
echo "  🎮 GPU Layers: $GPU_LAYERS"
echo "  📝 Log File: $LOG_FILE"
echo ""

# List the available models recursively in the specified directory
print_header "Scanning for available models in $MODELS_PATH..."
echo ""

# Function to format file size
format_size() {
    local size=$1
    if [ $size -ge 1073741824 ]; then
        echo "$(echo "scale=1; $size / 1073741824" | bc)GB"
    elif [ $size -ge 1048576 ]; then
        echo "$(echo "scale=1; $size / 1048576" | bc)MB"
    elif [ $size -ge 1024 ]; then
        echo "$(echo "scale=1; $size / 1024" | bc)KB"
    else
        echo "${size}B"
    fi
}

# Recursively search for all .gguf files and sort by size (largest first)
# Format: size|full_path
models_with_sizes=()
while IFS= read -r line; do
    models_with_sizes+=("$line")
done < <(find "$MODELS_PATH" -type f -name "*.gguf" -exec stat -c '%s|%n' {} \; 2>/dev/null | sort -t'|' -k1 -n -r)

# Check if any models were found
if [ ${#models_with_sizes[@]} -eq 0 ]; then
    print_error "No .gguf model files found in $MODELS_PATH"
    print_info "Please add .gguf model files to $MODELS_PATH or its subdirectories"
    exit 1
fi

print_success "Found ${#models_with_sizes[@]} models (sorted by size, largest first)"
echo ""

# Arrays to store model info
model_names=()
model_paths=()
model_sizes=()

# Parse the sorted results
for entry in "${models_with_sizes[@]}"; do
    size="${entry%%|*}"
    full_path="${entry#*|}"
    # Get relative path from MODELS_PATH
    relative_path="${full_path#$MODELS_PATH/}"
    
    model_names+=("$(basename "$full_path")")
    model_paths+=("$relative_path")
    model_sizes+=("$size")
done

# Display numbered list of models with file sizes and relative paths
for i in "${!model_names[@]}"; do
    formatted_size=$(format_size "${model_sizes[i]}")
    # Show the relative path for clarity
    echo -e "  $((i+1)). ${model_names[i]} (${YELLOW}$formatted_size${NC})"
    echo -e "      ${CYAN}${model_paths[i]}${NC}"
done

echo ""
print_info "Please select a model by entering its number (1-${#model_names[@]}):"
read -r selection

# Validate selection
if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt ${#model_names[@]} ]; then
    print_error "Invalid selection. Please run the script again and select a number between 1 and ${#model_names[@]}."
    exit 1
fi

# Store the selected model name and relative path for Docker
MODEL_NAME="${model_names[$((selection-1))]}"
MODEL_RELATIVE_PATH="${model_paths[$((selection-1))]}"
print_success "Selected model: $MODEL_RELATIVE_PATH"

# Ask user for run mode preference
echo ""
print_header "Server Run Mode Selection:"
echo "  1. 🖥️  Foreground mode (press Ctrl+C to stop)"
echo "  2. 🔄 Detached mode (runs in background)"
echo ""
print_info "Please select run mode (1-2):"
read -r run_mode

# Validate run mode selection
if ! [[ "$run_mode" =~ ^[0-9]+$ ]] || [ "$run_mode" -lt 1 ] || [ "$run_mode" -gt 2 ]; then
    print_error "Invalid selection. Please run the script again and select 1 or 2."
    exit 1
fi

# This script starts the server and waits for it to be ready.
echo ""
print_header "Preparing to start server with model: $MODEL_NAME"
echo ""

# Check if GPUs are available
print_info "Checking GPU availability..."
GPU_FLAG=""
if docker run --rm --gpus all "$CUDA_TEST_IMAGE" nvidia-smi >/dev/null 2>&1; then
    print_success "NVIDIA GPU detected and accessible."
    GPU_FLAG="--gpus all"
else
    print_warning "No GPU detected or NVIDIA Container Toolkit not installed."
    echo "Do you want to run the server in CPU-only mode? (y/n):"
    read -r cpu_choice
    if [[ "$cpu_choice" =~ ^[Yy]$ ]]; then
        print_info "Server will run in CPU-only mode."
        GPU_FLAG=""
    else
        print_error "Cannot proceed without GPU or CPU mode. Exiting."
        exit 1
    fi
fi
echo ""

# Check if Docker daemon is running
print_info "Checking Docker daemon status..."
if ! docker info >/dev/null 2>&1; then
    print_error "Docker daemon is not running."
    print_info "Attempting to start Docker daemon..."

    # Try to start Docker using systemctl (most common)
    if command -v systemctl >/dev/null 2>&1; then
        print_command "sudo systemctl start docker"
        sudo systemctl start docker
        if [ $? -eq 0 ]; then
            print_success "Docker daemon started successfully using systemctl."
            # Wait a moment for Docker to fully initialize
            print_info "Waiting 3 seconds for Docker to fully initialize..."
            sleep 3
        else
            print_error "Failed to start Docker daemon using systemctl."
            echo "Please start Docker manually and try again."
            exit 1
        fi
    # Try service command as fallback
    elif command -v service >/dev/null 2>&1; then
        print_command "sudo service docker start"
        sudo service docker start
        if [ $? -eq 0 ]; then
            print_success "Docker daemon started successfully using service command."
            sleep 3
        else
            print_error "Failed to start Docker daemon using service command."
            echo "Please start Docker manually and try again."
            exit 1
        fi
    else
        print_error "Cannot automatically start Docker daemon."
        print_info "Please start Docker manually using one of these commands:"
        print_command "  sudo systemctl start docker"
        print_command "  sudo service docker start"
        print_command "  sudo dockerd"
        exit 1
    fi

    # Verify Docker is now running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is still not accessible after start attempt."
        print_error "Please check Docker installation and permissions."
        exit 1
    fi
else
    print_success "Docker daemon is running."
fi

# Check if container already exists
if docker ps -a --format "table {{.Names}}" | grep -q "^llama-cpp-server$"; then
    print_warning "A container named 'llama-cpp-server' already exists."

    # Check if it's running
    if docker ps --format "table {{.Names}}" | grep -q "^llama-cpp-server$"; then
        print_info "The container is currently running."
        echo "Do you want to stop and remove it to start a new one? (y/n):"
    else
        print_info "The container is stopped."
        echo "Do you want to remove it to start a new one? (y/n):"
    fi

    read -r remove_choice

    if [[ "$remove_choice" =~ ^[Yy]$ ]]; then
        print_info "Stopping and removing existing container..."
        docker stop llama-cpp-server 2>/dev/null || true
        docker rm llama-cpp-server 2>/dev/null || true
        print_success "Existing container removed."
    else
        print_error "Cannot start new container. Exiting."
        exit 1
    fi
fi

echo ""
if [ "$run_mode" -eq 1 ]; then
    # Run in foreground mode
    print_header "Starting server in foreground mode (press Ctrl+C to stop)..."
    print_command "docker run --name llama-cpp-server $GPU_FLAG -v $MODELS_PATH:/models -p $SERVER_PORT:$SERVER_PORT $LLAMA_CPP_IMAGE --server -m /models/$MODEL_RELATIVE_PATH --host $HOST --port $SERVER_PORT -c $CONTEXT_SIZE -ngl $GPU_LAYERS"
    echo ""
    docker run --name llama-cpp-server $GPU_FLAG \
        -v $MODELS_PATH:/models -p $SERVER_PORT:$SERVER_PORT $LLAMA_CPP_IMAGE \
        --server -m /models/$MODEL_RELATIVE_PATH --host $HOST --port $SERVER_PORT -c $CONTEXT_SIZE -ngl $GPU_LAYERS
else
    # Run in detached mode
    print_header "Starting server in detached mode..."
    print_command "docker run -d --name llama-cpp-server $GPU_FLAG -v $MODELS_PATH:/models -p $SERVER_PORT:$SERVER_PORT $LLAMA_CPP_IMAGE --server -m /models/$MODEL_RELATIVE_PATH --host $HOST --port $SERVER_PORT -c $CONTEXT_SIZE -ngl $GPU_LAYERS"
    echo ""
    CONTAINER_ID=$(docker run -d $GPU_FLAG --name llama-cpp-server \
        -v $MODELS_PATH:/models -p $SERVER_PORT:$SERVER_PORT $LLAMA_CPP_IMAGE \
        --server -m /models/$MODEL_RELATIVE_PATH --host $HOST --port $SERVER_PORT -c $CONTEXT_SIZE -ngl $GPU_LAYERS)

    if [ $? -eq 0 ]; then
        print_success "Server started in background with container ID: $CONTAINER_ID"
        print_success "Server will be available at http://localhost:$SERVER_PORT"
        echo ""

        print_header "Useful Commands:"
        echo "  📊 Check server logs:"
        print_command "    docker logs llama-cpp-server"
        echo ""
        echo "  📊 Follow server logs:"
        print_command "    docker logs -f llama-cpp-server"
        echo ""
        echo "  ⏹️  Stop the server:"
        print_command "    docker stop llama-cpp-server"
        echo ""
        echo "  🗑️  Remove the container:"
        print_command "    docker rm llama-cpp-server"
    else
        print_error "Failed to start the server container."
        exit 1
    fi
fi

echo ""
print_success "Llama.cpp Docker server setup completed!"
echo -e "${PURPLE}============================================${NC}"
