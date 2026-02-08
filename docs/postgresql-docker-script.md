# PostgreSQL Docker Server Script

## Overview

`start-docker-postgresql-server.sh` is an interactive script that manages a PostgreSQL database server running in Docker. It provides a user-friendly way to configure, start, and manage a PostgreSQL container with pgvector extension support for vector similarity search.

## Purpose

This script automates the setup and management of a PostgreSQL development database with the following capabilities:
- **Vector Search**: Includes pgvector extension for embeddings and similarity search
- **Flexible Configuration**: Supports environment variables, .env files, and interactive prompts
- **Container Management**: Handles Docker image selection, pulling, and container lifecycle
- **User Creation**: Automatically creates databases and users if they don't exist
- **Security**: Hides password input from terminal display

## Basic Usage

### Quick Start

Simply run the script to start with interactive configuration:

```bash
./start-docker-postgresql-server.sh
```

The script will:
1. Prompt to load environment variables from `.env` (if present)
2. Ask for any missing configuration values
3. Start or create the PostgreSQL Docker container
4. Set up the database, user, and pgvector extension

### Using Environment Variables

Configure the database by setting environment variables before running:

```bash
POSTGRES_DB=myapp POSTGRES_USER=admin ./start-docker-postgresql-server.sh
```

### Command-Line Arguments

View all available options:

```bash
./start-docker-postgresql-server.sh --help
```

## Configuration

### Environment Variables

The script recognizes these environment variables (with defaults):

- `POSTGRES_DB` - Database name (default: dev)
- `POSTGRES_USER` - Database user (default: dev)
- `POSTGRES_PASSWORD` - User password (prompted if not set)
- `POSTGRES_PORT` - Host port mapping (default: 5432)
- `POSTGRES_HOST` - Database host (default: localhost)
- `PGDATA` - Data volume path (default: postgres_data)
- `POSTGRES_IMAGE` - Docker image (default: pgvector/pgvector:pg15)

### Configuration Priority

Settings are applied in this order (highest priority first):
1. Command-line arguments
2. Environment variables
3. Values from `.env` file
4. Interactive user input
5. Built-in defaults

## Features

### pgvector Extension

The default image includes the pgvector extension for storing and querying vector embeddings. This is essential for:
- Semantic search
- RAG (Retrieval-Augmented Generation) applications
- Similarity-based queries
- Machine learning feature storage

### Automatic Setup

The script automatically:
- Pulls the Docker image if not present
- Creates the container with proper configuration
- Initializes the database and user
- Enables the pgvector extension
- Handles existing containers gracefully

### Password Security

User passwords are never displayed on screen:
- Interactive password entry uses hidden input (`read -rs`)
- Passwords are securely passed to container environment variables
- No passwords stored in command history

## Container Details

- **Container Name**: `postgres-db`
- **Default Image**: `pgvector/pgvector:pg15`
- **PostgreSQL Version**: 15
- **Data Persistence**: Uses Docker volume for data storage
- **Network**: Exposed to host on configured port

## Common Use Cases

### Development Database

Start a local development database with default settings:
```bash
./start-docker-postgresql-server.sh
```

### Custom Configuration

Use a specific database and user:
```bash
POSTGRES_DB=myproject POSTGRES_USER=myuser ./start-docker-postgresql-server.sh
```

### Production-Like Setup

Configure with strong credentials and custom port:
```bash
POSTGRES_USER=admin POSTGRES_PORT=5433 ./start-docker-postgresql-server.sh
```

## Makefile Integration

This script is integrated with the project Makefile:
- `make db-start` - Runs this script
- `make db-connect` - Connects to the database
- `make db-stop` - Stops the container
- `make db-help` - Shows this script's help

See [`docs/makefile-usage.md`](./makefile-usage.md) for more details.

## Troubleshooting

### Port Already in Use

If port 5432 is already in use, specify a different port:
```bash
POSTGRES_PORT=5433 ./start-docker-postgresql-server.sh
```

### Container Already Exists

The script will detect existing containers and prompt you to remove or keep them.

### Docker Not Running

The script will attempt to start Docker daemon if it's not running (requires sudo).

### Collation Version Warnings

After PostgreSQL upgrades, you may see collation warnings. Fix with:
```sql
ALTER DATABASE your_database REFRESH COLLATION VERSION;
```

## Future Enhancements

The script may be extended to support:
- Multiple database instances
- Replication and clustering setup
- Backup and restore automation
- Custom PostgreSQL extensions
- Performance monitoring integration
- Connection pooling (PgBouncer)
- SSL/TLS configuration
- Custom initialization scripts
- Migration tool integration

## Related Documentation

- [Makefile Usage](./makefile-usage.md)
- [GPU Setup](./gpu-setup.md)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Docker Official Images](https://hub.docker.com/_/postgres)
