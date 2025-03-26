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
- **OpenAI-compatible API**: Drop-in replacement for OpenAI's chat completions and embeddings API
- **Ollama Models Support**: Can use existing Ollama models repository without re-downloading
- **Intelligent Token Truncation**: Automatically truncates embedding inputs according to model context size
- **Robust Health Checking**: Ensures containers are truly ready to accept requests before serving traffic

## Requirements

- Python 3.8+
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU(s) with CUDA support
- Ollama models repository (automatically downloaded if not provided)

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

3. Configure your models in `config.yaml`. Several key points:
   - Set `debug` to `true` when debugging (debug port: 5001). When `debug=false`, the port is 5000
   - Set the path to your Ollama models repository (necessary if you want to re-use Ollama models)
   - Set the HF_path to your HuggingFace cache path (necessary if you want to re-use locally downloaded models from HuggingFace)
   - Set model-specific parameters
     - **Naming**: Use the Ollama model format: `[namespace/]name[:tag]` (e.g., `qwen2.5:14b`, `llama3:8b`)
     - **Embedding**: Set `model_ctx_size` properly to match the tokenizer's context window. Need to check the official docs of the model you are using (`important and necessary`); Or, the truncation won't work and it will raise error. Note: Embedding models don't benefit much from `parallel`, `flash_attn`, or `cont_batching` settings. Set `parallel` to 1 and turn off `flash_attn` and `cont_batching`.


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
   # Chat completions example
   curl http://localhost:5000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen2.5:14b",
       "messages": [{"role": "user", "content": "Hello, how are you?"}],
       "temperature": 0.7
     }'
     
   # Embeddings example
   curl http://localhost:5000/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{
       "model": "all-minilm:latest",
       "input": "Hello, I would like to get the embedding for this text"
     }'
   ```

## API Endpoints

- **`/v1/chat/completions`**: OpenAI-compatible chat completions API
- **`/api/chat`**: OpenAI-compatible chat completions API
- **`/v1/embeddings`**: OpenAI-compatible embeddings API
- **`/api/embed`**: OpenAI-compatible embeddings API
- **`/dashboard`**: Web dashboard showing available and running models

## License

MIT

