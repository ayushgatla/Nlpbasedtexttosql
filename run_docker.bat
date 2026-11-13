@echo off
echo ===================================================
echo Starting NeuroSQL Studio via Docker Compose
echo ===================================================

where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not found in PATH.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo Building and starting containers in detached mode...
docker compose up --build -d

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] NeuroSQL Studio is running!
    echo Access the visual studio at: http://localhost:8501
    echo To view logs, run: docker compose logs -f
    echo To stop the container, run: docker compose down
) else (
    echo.
    echo [ERROR] Failed to start Docker container.
)
