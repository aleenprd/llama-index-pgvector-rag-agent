.PHONY: help db-start db-stop db-restart db-connect db-logs db-logs-follow db-status db-remove db-remove-all db-shell db-help llama-start llama-stop llama-restart llama-logs llama-logs-follow llama-status llama-remove llama-help

# Load environment variables from .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Default target
help:
	@echo "🚀 Docker Management Commands"
	@echo ""
	@echo "═══════════════════════════════════════════════"
	@echo "🐘 PostgreSQL Database"
	@echo "═══════════════════════════════════════════════"
	@echo "Setup:"
	@echo "  make db-start         - Start PostgreSQL container"
	@echo "  make db-stop          - Stop PostgreSQL container"
	@echo "  make db-restart       - Restart PostgreSQL container"
	@echo ""
	@echo "Connect:"
	@echo "  make db-connect       - Connect to database using .env variables"
	@echo "  make db-shell         - Open bash shell in container"
	@echo ""
	@echo "Monitor:"
	@echo "  make db-logs          - View container logs"
	@echo "  make db-logs-follow   - Follow container logs (live)"
	@echo "  make db-status        - Show container status"
	@echo ""
	@echo "Cleanup:"
	@echo "  make db-remove        - Remove container (keeps data)"
	@echo "  make db-remove-all    - Remove container AND volume (⚠️  loses ALL data)"
	@echo ""
	@echo "Help:"
	@echo "  make db-help          - Show detailed PostgreSQL server script help"
	@echo ""
	@echo "═══════════════════════════════════════════════"
	@echo "🦙 Llama.cpp Server"
	@echo "═══════════════════════════════════════════════"
	@echo "Setup:"
	@echo "  make llama-start      - Start Llama.cpp server (interactive)"
	@echo "  make llama-stop       - Stop Llama.cpp server"
	@echo "  make llama-restart    - Restart Llama.cpp server"
	@echo ""
	@echo "Monitor:"
	@echo "  make llama-logs       - View server logs"
	@echo "  make llama-logs-follow - Follow server logs (live)"
	@echo "  make llama-status     - Show server status"
	@echo ""
	@echo "Cleanup:"
	@echo "  make llama-remove     - Remove server container"
	@echo ""
	@echo "Help:"
	@echo "  make llama-help       - Show detailed Llama.cpp server script help"
	@echo ""
	@echo "💡 Tip: Run 'make db-help' or 'make llama-help' for configuration options"

# Show detailed help from the PostgreSQL server script
db-help:
	@./start-docker-postgresql-server.sh --help

# Start the PostgreSQL container using the shell script
db-start:
	@./start-docker-postgresql-server.sh

# Stop the container
db-stop:
	@echo "⏹️  Stopping PostgreSQL container..."
	@docker stop postgres-db

# Restart the container
db-restart: db-stop
	@echo "▶️  Starting PostgreSQL container..."
	@docker start postgres-db

# Connect to database using environment variables
db-connect:
	@psql -h $${POSTGRES_HOST} -p $${POSTGRES_PORT:-5432} -U $${POSTGRES_USER} -d $${POSTGRES_DB}

# Open bash shell in the container
db-shell:
	@docker exec -it postgres-db bash

# View container logs
db-logs:
	@docker logs postgres-db

# Follow container logs (live)
db-logs-follow:
	@docker logs -f postgres-db

# Show container status
db-status:
	@docker ps --filter "name=postgres-db" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"

# Remove container (keeps volume/data)
db-remove:
	@echo "🗑️  Removing PostgreSQL container..."
	@docker stop postgres-db 2>/dev/null || true
	@docker rm postgres-db 2>/dev/null || true
	@echo "✅ Container removed (data preserved in volume)"

# Remove container AND volume (loses all data)
db-remove-all:
	@echo "⚠️  WARNING: This will remove ALL PostgreSQL data!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read confirm
	@docker stop postgres-db 2>/dev/null || true
	@docker rm postgres-db 2>/dev/null || true
	@docker volume rm postgres_data 2>/dev/null || true
	@echo "✅ Container and data volume removed"

# ═══════════════════════════════════════════════
# Llama.cpp Server Commands
# ═══════════════════════════════════════════════

# Show detailed help from the Llama.cpp server script
llama-help:
	@./start-docker-llamacpp-server.sh --help

# Start the Llama.cpp server using the shell script (interactive)
llama-start:
	@./start-docker-llamacpp-server.sh

# Stop the server
llama-stop:
	@echo "⏹️  Stopping Llama.cpp server..."
	@docker stop llama-cpp-server

# Restart the server
llama-restart: llama-stop
	@echo "▶️  Starting Llama.cpp server..."
	@docker start llama-cpp-server

# View server logs
llama-logs:
	@docker logs llama-cpp-server

# Follow server logs (live)
llama-logs-follow:
	@docker logs -f llama-cpp-server

# Show server status
llama-status:
	@docker ps --filter "name=llama-cpp-server" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"

# Remove server container
llama-remove:
	@echo "🗑️  Removing Llama.cpp server container..."
	@docker stop llama-cpp-server 2>/dev/null || true
	@docker rm llama-cpp-server 2>/dev/null || true
	@echo "✅ Container removed"
