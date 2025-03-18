# Llama.cpp Scheduler

A dynamic scheduler for managing multiple Llama.cpp model instances in Docker containers. A lightweight alternative to Ollama.

## Why not Ollama?

While Ollama provides a user-friendly interface for running LLMs, I met a significant issue where it gets stuck after a few runs, especially under concurrency. This is a well-documented problem (see [Ollama issue #1863](https://github.com/ollama/ollama/issues/1863)) that persists even in recent versions (tested in v0.5.13).

## Features

- **On-demand Model Loading**: Models are only downloaded and loaded when requested
- **Automatic Resource Management**: Selects the best available GPU for each model
- **Container Lifecycle Management**: Automatically starts and stops containers as needed
- **Inactivity Monitoring**: Shuts down idle containers to free up resources
- **Web Dashboard**: Monitor running models and their statistics
- **OpenAI-compatible API**: Drop-in replacement for OpenAI's chat completions API (embedding API not supported yet)

## Requirements

- Python 3.8+
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU(s) with CUDA support
- `axel` for parallel downloads (optional but recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llama-cpp-scheduler.git
   cd llama-cpp-scheduler
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your models in `config.yaml`. You need to find the download url for the model you want to use from the model providers such as huggingface, modelscope, etc. The gguf file should be in a single file. Also, remember to set chat-template for each model.

## Usage

1. Start the scheduler (you need root access to start docker containers):
   ```bash
   ./run.sh
   ```

2. Access the web dashboard:
   ```
   http://localhost:5000/dashboard
   ```

3. Send requests to the API:
   ```bash
   curl http://localhost:5000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen2.5-14b",
       "messages": [{"role": "user", "content": "Hello, how are you?"}],
       "temperature": 0.7
     }'
   ```

## API Endpoints

- **`/v1/chat/completions`**: OpenAI-compatible chat completions API
- other openai compatible endpoints to be added soon
- **`/dashboard`**: Web dashboard showing available and running models


## How It Works

1. When a request is received, the scheduler checks if the requested model is downloaded
2. If not downloaded, it downloads the model from the configured URL
3. The scheduler then starts a Docker container for the model if not already running
4. The request is forwarded to the container
5. The container is automatically stopped after a period of inactivity

## License

MIT

