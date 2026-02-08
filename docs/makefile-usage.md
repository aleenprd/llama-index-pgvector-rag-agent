# Makefile Usage Guide

## Overview

The Makefile provides convenient shortcuts for common development tasks in this project. It acts as a command runner and task automation tool, making complex commands easier to remember and execute.

## Philosophy

The Makefile follows these principles:
- **Environment-aware**: Loads configuration from `.env` when available
- **User-friendly**: Provides clear, memorable command names
- **Self-documenting**: Includes help text for all commands
- **Flexible**: Uses environment variables with sensible defaults
- **Evolving**: Commands and functionality may change as project needs grow

## General Usage

```bash
make <command>
```

To see all available commands:
```bash
make help
```

## Current Command Categories

### Database Management

Commands for managing the PostgreSQL Docker container:

- **Setup**: Start, stop, and restart the database
- **Connect**: Access the database shell
- **Monitor**: View logs and status
- **Cleanup**: Remove containers and data

Run `make help` to see the complete list of database commands.

### Environment Variables

The Makefile automatically sources environment variables from `.env` if the file exists. This allows you to configure behavior without modifying the Makefile itself.

Common variables used:
- `POSTGRES_HOST` - Database host (default: localhost)
- `POSTGRES_PORT` - Database port (default: 5432)
- `POSTGRES_USER` - Database user (default: dev)
- `POSTGRES_DB` - Database name (default: dev)
- `POSTGRES_PASSWORD` - Database password (prompted when needed)

## Adding New Commands

As the project grows, new commands can be added to automate:
- Running tests
- Building and deploying applications
- Managing other services (Redis, message queues, etc.)
- Code formatting and linting
- Data migrations
- Development server management

## Best Practices

1. **Check help first**: Run `make help` to see available commands
2. **Use .env**: Store configuration in `.env` rather than hardcoding
3. **Read command output**: Commands provide feedback about what they're doing
4. **Understand before running**: Some commands (like `db-remove-all`) are destructive

## Why Makefile?

While traditionally used for building software, Makefiles are excellent task runners because:
- **Ubiquitous**: Available on all Unix-like systems by default
- **Simple syntax**: Easy to read and modify
- **Dependency management**: Can chain commands and define prerequisites
- **Standard tool**: No additional installation required
- **IDE support**: Most editors understand Makefile syntax

## Alternative Approaches

This project uses Make for convenience, but you can also:
- Run the underlying shell scripts directly (e.g., `./start-docker-postgresql-server.sh`)
- Use Docker and psql commands directly
- Execute Python scripts with the virtual environment activated
- Set up your own task runner (npm scripts, just, task, etc.)

## Future Directions

The Makefile may be extended to include:
- Application build and run commands
- Test execution and coverage reporting
- Container orchestration for multi-service setups
- CI/CD pipeline integration
- Development environment setup automation
- Code quality checks and formatting

Check `make help` periodically as new commands are added to support evolving project needs.
