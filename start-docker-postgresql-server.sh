#!/bin/bash

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
    echo -e "${PURPLE}[POSTGRESQL]${NC} $1"
}

print_command() {
    echo -e "${CYAN}[COMMAND]${NC} $1"
}

# Function to show help
show_help() {
    echo -e "${PURPLE}============================================${NC}"
    echo -e "${PURPLE}    PostgreSQL Docker Server Manager       ${NC}"
    echo -e "${PURPLE}============================================${NC}"
    echo ""
    echo "Usage: ./start-docker-postgresql-server.sh [OPTIONS]"
    echo ""
    echo "Description:"
    echo "  Manages a PostgreSQL 15 Docker container with pgvector extension and persistent storage."
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Environment Variables (can be set in .env file):"
    echo "  POSTGRES_IMAGE     Docker image to use (default: pgvector/pgvector:pg15)"
    echo "  POSTGRES_DB        Database name (default: mydb)"
    echo "  POSTGRES_USER      Database user (default: myuser)"
    echo "  POSTGRES_PASSWORD  Database password (default: mypassword)"
    echo "  POSTGRES_PORT      PostgreSQL port (default: 5432)"
    echo "  POSTGRES_HOST      PostgreSQL host (default: localhost)"
    echo "  PGDATA             Data directory path (default: /var/lib/postgresql/data)"
    echo ""
    echo "Quick Start:"
    echo "  1. Run the script: ./start-docker-postgresql-server.sh"
    echo "  2. Connect: psql -h localhost -U myuser -d mydb"
    echo ""
    echo "Examples:"
    echo "  # Start with defaults"
    echo "  ./start-docker-postgresql-server.sh"
    echo ""
    echo "  # Use environment variables"
    echo "  POSTGRES_USER=admin ./start-docker-postgresql-server.sh"
    echo ""
    echo "  # Or create a .env file with your settings"
    echo "  echo 'POSTGRES_USER=admin' > .env"
    echo "  ./start-docker-postgresql-server.sh"
    echo ""
    echo "For more commands, see the Makefile:"
    echo "  make help"
    echo ""
    exit 0
}

# Check for help flag
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
fi

# Print script header
echo -e "${PURPLE}============================================${NC}"
echo -e "${PURPLE}    PostgreSQL Docker Server Manager       ${NC}"
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
        print_info "Available environment variables:"
        echo "  - POSTGRES_DB: ${POSTGRES_DB:-mydb} (default: mydb)"
        echo "  - POSTGRES_USER: ${POSTGRES_USER:-myuser} (default: myuser)"
        echo "  - POSTGRES_PASSWORD: ******** (default: mypassword)"
        echo "  - POSTGRES_PORT: ${POSTGRES_PORT:-5432} (default: 5432)"
        echo "  - POSTGRES_HOST: ${POSTGRES_HOST:-localhost} (default: localhost)"
        echo "  - POSTGRES_IMAGE: ${POSTGRES_IMAGE:-pgvector/pgvector:pg15} (default: pgvector/pgvector:pg15)"
        echo "  - PGDATA: ${PGDATA:-/var/lib/postgresql/data} (default: /var/lib/postgresql/data)"
    else
        print_warning "Skipping .env file."
    fi
    echo ""
fi

# If environment variables are not set, ask user for input
if [ -z "$POSTGRES_DB" ] || [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ]; then
    print_info "Configuration not found in environment variables."
    echo "Would you like to configure the database settings manually? (y/n):"
    read -r manual_config
    
    if [[ "$manual_config" =~ ^[Yy]$ ]]; then
        echo ""
        print_header "Database Configuration"
        
        # Main database name
        echo -n "Enter database name [default: mydb]: "
        read -r user_db
        POSTGRES_DB=${user_db:-mydb}
        
        # Main database user
        echo -n "Enter database username [default: myuser]: "
        read -r user_name
        POSTGRES_USER=${user_name:-myuser}
        
        # Main database password
        echo -n "Enter database password [default: mypassword]: "
        read -rs user_pass
        echo ""
        POSTGRES_PASSWORD=${user_pass:-mypassword}
        
        # Port
        echo -n "Enter PostgreSQL port [default: 5432]: "
        read -r user_port
        POSTGRES_PORT=${user_port:-5432}
        
        # Host
        echo -n "Enter PostgreSQL host [default: localhost]: "
        read -r user_host
        POSTGRES_HOST=${user_host:-localhost}

        # Data directory
        echo -n "Enter data directory path [default: /var/lib/postgresql/data]: "
        read -r user_data
        PGDATA=${user_data:-/var/lib/postgresql/data}

        # Docker image
        echo -n "Enter PostgreSQL Docker image [default: pgvector/pgvector:pg15]: "
        read -r user_image
        POSTGRES_IMAGE=${user_image:-pgvector/pgvector:pg15}

        echo ""
        print_success "Configuration completed!"
        print_info "Database: ${POSTGRES_DB}"
        print_info "User: ${POSTGRES_USER}"
        print_info "Port: ${POSTGRES_PORT}"
        print_info "Host: ${POSTGRES_HOST}"
        print_info "Data Directory: ${PGDATA}"
        print_info "Docker Image: ${POSTGRES_IMAGE}"
        echo ""
    else
        print_info "Using default values."
        POSTGRES_DB=${POSTGRES_DB:-mydb}
        POSTGRES_USER=${POSTGRES_USER:-myuser}
        POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-mypassword}
        POSTGRES_PORT=${POSTGRES_PORT:-5432}
        POSTGRES_HOST=${POSTGRES_HOST:-localhost}
        PGDATA=${PGDATA:-/var/lib/postgresql/data}
        POSTGRES_IMAGE=${POSTGRES_IMAGE:-pgvector/pgvector:pg15}
    fi
else
    # Use environment variables or defaults
    POSTGRES_DB=${POSTGRES_DB:-mydb}
    POSTGRES_USER=${POSTGRES_USER:-myuser}
    POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-mypassword}
    POSTGRES_PORT=${POSTGRES_PORT:-5432}
    POSTGRES_HOST=${POSTGRES_HOST:-localhost}
    PGDATA=${PGDATA:-/var/lib/postgresql/data}
    POSTGRES_IMAGE=${POSTGRES_IMAGE:-pgvector/pgvector:pg15}
fi

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
            print_info "Please start Docker manually and try again."
            exit 1
        fi
    # Try service command as fallback
    elif command -v service >/dev/null 2>&1; then
        print_command "sudo service docker start"
        sudo service docker start
        if [ $? -eq 0 ]; then
            print_success "Docker daemon started successfully using service command."
            print_info "Waiting 3 seconds for Docker to fully initialize..."
            sleep 3
        else
            print_error "Failed to start Docker daemon using service command."
            print_info "Please start Docker manually and try again."
            exit 1
        fi
    else
        print_error "Cannot automatically start Docker daemon."
        print_info "Please start Docker manually using one of these commands:"
        echo "  sudo systemctl start docker"
        echo "  sudo service docker start"
        echo "  sudo dockerd"
        exit 1
    fi

    # Verify Docker is now running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is still not accessible after start attempt."
        print_info "Please check Docker installation and permissions."
        exit 1
    fi
else
    print_success "Docker daemon is running."
fi

# Check if container already exists
print_info "Checking for existing PostgreSQL container..."
if docker ps -a --format "table {{.Names}}" | grep -q "^postgres-db$"; then
    print_warning "A container named 'postgres-db' already exists."

    # Check if it's running
    if docker ps --format "table {{.Names}}" | grep -q "^postgres-db$"; then
        print_info "The container is currently running."
        print_info "Container status:"
        docker ps --filter "name=postgres-db" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        echo "Do you want to stop and remove it to start a new one? (y/n):"
    else
        print_warning "The container is stopped."
        print_info "Container status:"
        docker ps -a --filter "name=postgres-db" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        echo "Do you want to remove it to start a new one? (y/n):"
    fi

    read -r remove_choice

    if [[ "$remove_choice" =~ ^[Yy]$ ]]; then
        print_info "Stopping and removing existing container..."

        # Stop container if running
        if docker ps --format "table {{.Names}}" | grep -q "^postgres-db$"; then
            print_command "docker stop postgres-db"
            docker stop postgres-db 2>/dev/null || true
            print_success "Container stopped."
        fi

        # Remove container
        print_command "docker rm postgres-db"
        docker rm postgres-db 2>/dev/null || true
        print_success "Existing container removed."
    else
        print_warning "Cannot start new container while existing one exists. Exiting."
        print_info "If you want to connect to the existing container, use:"
        print_command "docker exec -it postgres-db psql -U ${DB_USER:-myuser} -d ${DB_NAME:-mydb}"
        exit 1
    fi
else
    print_success "No existing PostgreSQL container found. Ready to create new one."
fi

echo ""
print_header "Starting PostgreSQL container with the following configuration:"
echo "  🗃️  Database Name: ${POSTGRES_DB:-mydb}"
echo "  👤 Username: ${POSTGRES_USER:-myuser}"
echo "  🔑 Password: ********"
echo "  🌐 Port: ${POSTGRES_PORT:-5432}"
echo "  🌐 Host: ${POSTGRES_HOST:-localhost}"
echo "  🐳 Docker Image: ${POSTGRES_IMAGE:-pgvector/pgvector:pg15}"
echo "  📁 Data Dir: ${PGDATA:-/var/lib/postgresql/data}"
echo ""

# Check if postgres image exists locally
print_info "Checking for Docker image: ${POSTGRES_IMAGE}..."
if ! docker images | grep -q "${POSTGRES_IMAGE%%:*}.*${POSTGRES_IMAGE##*:}"; then
    print_warning "Docker image ${POSTGRES_IMAGE} not found locally."
    print_info "Pulling ${POSTGRES_IMAGE} from Docker Hub..."
    print_command "docker pull ${POSTGRES_IMAGE}"
    docker pull "${POSTGRES_IMAGE}"
    if [ $? -eq 0 ]; then
        print_success "Docker image ${POSTGRES_IMAGE} downloaded successfully!"
    else
        print_error "Failed to pull Docker image ${POSTGRES_IMAGE}."
        exit 1
    fi
else
    print_success "Docker image ${POSTGRES_IMAGE} found."
fi

# With data persistence (recommended)
print_info "Creating PostgreSQL container with data persistence..."
print_command "docker run --name postgres-db \
    -e POSTGRES_DB=\"${POSTGRES_DB:-mydb}\" \
    -e POSTGRES_USER=\"${POSTGRES_USER:-myuser}\" \
    -e POSTGRES_PASSWORD=\"********\" \
    -p \"${POSTGRES_PORT:-5432}:5432\" \
    -v \"postgres_data:${PGDATA:-/var/lib/postgresql/data}\" \
    -d \"${POSTGRES_IMAGE}\""

docker run --name postgres-db \
    -e POSTGRES_DB="${POSTGRES_DB:-mydb}" \
    -e POSTGRES_USER="${POSTGRES_USER:-myuser}" \
    -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-mypassword}" \
    -p "${POSTGRES_PORT:-5432}:5432" \
    -v "postgres_data:${PGDATA:-/var/lib/postgresql/data}" \
    -d "${POSTGRES_IMAGE}"

if [ $? -eq 0 ]; then
    print_success "PostgreSQL container started successfully!"

    print_info "Waiting 5 seconds for PostgreSQL to initialize..."
    sleep 5

    # Check if container is actually running
    if docker ps --format "table {{.Names}}" | grep -q "^postgres-db$"; then
        print_success "Container is running and healthy."

        print_header "Connection Information:"
        echo "  🌐 Host: ${POSTGRES_HOST:-localhost}"
        echo "  🔌 Port: ${POSTGRES_PORT:-5432}"
        echo "  🗃️  Database: ${POSTGRES_DB:-mydb}"
        echo "  👤 Username: ${POSTGRES_USER:-myuser}"
        echo "  🔑 Password: ********"
        echo ""
        echo "  📎 Connection String:"
        echo "     postgresql://${POSTGRES_USER:-myuser}:********@${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-mydb}"
        echo ""

        print_header "Useful Commands:"
        echo "  🔗 Connect to database:"
        print_command "    docker exec -it postgres-db psql -U ${POSTGRES_USER:-myuser} -d ${POSTGRES_DB:-mydb}"
        echo ""
        echo "  📊 View container logs:"
        print_command "    docker logs postgres-db"
        echo ""
        echo "  📊 Follow container logs:"
        print_command "    docker logs -f postgres-db"
        echo ""
        echo "  ⏹️  Stop container:"
        print_command "    docker stop postgres-db"
        echo ""
        echo "  ▶️  Start stopped container:"
        print_command "    docker start postgres-db"
        echo ""
        echo "  🗑️  Remove container (⚠️  will lose data if no volume):"
        print_command "    docker stop postgres-db && docker rm postgres-db"
        echo ""
        echo "  🗑️  Remove container AND volume (⚠️  will lose ALL data):"
        print_command "    docker stop postgres-db && docker rm postgres-db && docker volume rm postgres_data"
        echo ""

        print_header "Container Status:"
        docker ps --filter "name=postgres-db" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"

    else
        print_error "Container failed to start properly."
        print_info "Checking container logs for errors..."
        docker logs postgres-db
    fi
else
    print_error "Failed to start PostgreSQL container."
    print_info "Please check Docker daemon status and try again."
    exit 1
fi

echo ""
print_success "PostgreSQL Docker server setup completed!"
echo -e "${PURPLE}============================================${NC}"