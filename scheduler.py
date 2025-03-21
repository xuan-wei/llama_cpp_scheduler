import time
from threading import Thread, Lock
from flask import Flask, request, jsonify, Response
import requests
import json
import docker
import signal
import sys
import yaml
from utils import (get_best_gpu, find_next_available_port, ensure_models_exist, 
                   is_container_running, get_container, setup_logger, is_ollama_model_available, find_ollama_model_file)
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timedelta
import os
from transformers import AutoTokenizer
import torch
import subprocess

logger = setup_logger('scheduler')

app = Flask(__name__)

# Docker client
docker_client = docker.from_env()  # Run as root

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update global variables from config
OLLAMA_PATH = config.get('ollama_path', '')
HF_PATH = config.get('HF_path', '')
HF_MIRROR = config.get('HF_MIRROR', '')
DOCKER_IMAGE = config['docker_image']
INACTIVITY_TIMEOUT = config['inactivity_timeout']
INACTIVITY_CHECK_INTERVAL = config['inactivity_check_interval']
CONTAINER_STARTUP_TIMEOUT = config['container_startup_timeout']
HEALTH_CHECK_INTERVAL = config['health_check_interval']
FORCE_GPU_ID = config['force_gpu_id']
REQUEST_TIMEOUT = config['request_timeout']

docker_name_prefix = config['docker_name_prefix']
if config['debug']:
    docker_name_prefix = docker_name_prefix + '_debug'

# Initialize model configurations from config file
model_configs = {}
for model_name, model_config in config['models'].items():
    model_configs[model_name] = {
        **config['common_config'],
        **model_config,
        "container_name": docker_name_prefix + '_' + model_name.replace('/', '_').replace(':', '-')
    }

# Create a mapping of alternative names to primary model names
model_name_mapping = {}
for model_name, model_config in model_configs.items():
    if 'other_names' in model_config:
        for alt_name in model_config['other_names']:
            model_name_mapping[alt_name] = model_name

# Dictionary to track the last request time for each model
model_last_request = {}
model_lock = Lock()  # Lock for thread-safe operations on the model_last_request dict

# Container operation locks to prevent multiple simultaneous start operations
container_locks = {}
container_locks_lock = Lock()  # Meta-lock to protect the container_locks dictionary

# embedding model prefix length
embedding_model_prefix_length = {}
embedding_model_prefix_length_lock = Lock()

# Initialize last request time for all models
for model_name in model_configs:
    model_last_request[model_name] = time.time()

# Statistics tracking
model_stats = {}
model_stats_lock = Lock()

# Add a tokenizer cache to avoid loading tokenizers repeatedly
tokenizer_cache = {}
tokenizer_cache_lock = Lock()  # Lock for thread-safe operations on the tokenizer cache

# Add a lock for tokenizer operations
tokenizer_operation_lock = Lock()

def get_container_lock(container_name):
    """Get or create a lock for a specific container."""
    with container_locks_lock:
        if container_name not in container_locks:
            container_locks[container_name] = Lock()
        return container_locks[container_name]

def start_container(model_name, service_type, model_file_path=None):
    """Start a Docker container with the specified parameters using Docker SDK."""
    if model_name not in model_configs:
        logger.error(f"Model {model_name} not found in configuration")
        return False
    
    config = model_configs[model_name]
    container_name = config["container_name"]
    
    try:
        # Check if container is already running
        if is_container_running(model_configs[model_name]["port"], container_name):
            logger.info(f"Container {container_name} is already running")
            return True
        
        # Check if container exists but is stopped
        existing_container = get_container(container_name, docker_client)
        if existing_container:
            logger.info(f"Found existing container {container_name} in state: {existing_container.status}")
            try:
                logger.info(f"Removing existing container {container_name}")
                existing_container.remove(force=True)
                logger.info(f"Successfully removed existing container {container_name}")
            except docker.errors.APIError as e:
                logger.error(f"Error removing existing container {container_name}: {str(e)}")
                return False
        
        # Get the best GPU to use
        config["device_id"] = get_best_gpu(FORCE_GPU_ID)
        logger.info(f"Selected GPU {config['device_id']} for model {model_name}")
        
        # If no port is assigned, find the next available one
        config["port"] = find_next_available_port(start_port=8090)
        logger.info(f"Assigned port {config['port']} to model {model_name}")
        
        # Prepare environment variables and command
        environment = {
            "CUDA_VISIBLE_DEVICES": str(config["device_id"]),
        }
        if HF_MIRROR:
            environment["HF_MIRROR"] = HF_MIRROR
        
        # Prepare port mapping
        ports = {f"{config['port']}/tcp": config['port']}
                
        # Prepare command
        command = [
            "--model", model_file_path,
            "--n-gpu-layers", str(config["n_gpu_layers"]),
            "--host", "0.0.0.0",
            "--port", str(config["port"]),
            "--threads", str(config["threads"]),  
        ]
        if config["cont_batching"]:
            command.extend(["--cont-batching"])
        if config["flash_attn"]:
            command.extend(["--flash-attn"])
        
        command.extend(["--ctx-size", str(config["ctx_size"])])
        command.extend(["--parallel", str(config["parallel"])])

        if service_type == 'chat':          
            command.extend(["--predict", str(config["predict"])])
            command.extend(["--ctx-size", str(config["ctx_size"])])
            command.extend(["--parallel", str(config["parallel"])])
            if config.get("chat_template"):
                command.extend(["--chat-template", config["chat_template"]])

        if service_type == 'embedding':          
            # Add embedding-specific parameters
            command.extend(["--embedding",
                            "--batch-size", str(config["model_ctx_size"]),
                            "--ubatch-size", str(config["model_ctx_size"])])
            if config.get("rope_scaling"):
                command.extend(["--rope-scaling", config["rope_scaling"]])
            if config.get("rope_freq_scale"):
                command.extend(["--rope-freq-scale", config["rope_freq_scale"]])
            
        volumes = {}
        
        # Mount Ollama repository
        if OLLAMA_PATH and os.path.isdir(OLLAMA_PATH):
            logger.info(f"Mounting Ollama repository: {OLLAMA_PATH}")
            volumes[OLLAMA_PATH] = {'bind': '/ollama_models', 'mode': 'ro'}
            
            # Adjust the model path in the command
            ollama_rel_path = os.path.relpath(model_file_path, OLLAMA_PATH)
            command[1] = f"/ollama_models/{ollama_rel_path}"
            logger.info(f"Adjusted model path: {command[1]}")
        
        # Log the full command for debugging
        logger.info(f"Starting container {container_name} on device {config['device_id']} with command: {' '.join(command)}")
        
        # Create and start the container
        container = docker_client.containers.run(
            DOCKER_IMAGE,
            command=command,
            name=container_name,
            detach=True,
            environment=environment,
            ports=ports,
            volumes=volumes,
            runtime="nvidia"  # Remove if not using NVIDIA GPU
        )
        
        logger.info(f"Container {container_name} started with ID: {container.id}")
        
        # Wait for the container to be ready
        if not wait_for_container_ready(model_name):
            logger.error(f"Container {container_name} failed to become ready")
            return False
        
        return True
        
    except docker.errors.APIError as e:
        logger.error(f"Error starting container {container_name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error starting container {container_name}: {str(e)}")
        return False

def stop_container(model_name):
    """Stop a Docker container using Docker SDK."""
    if model_name not in model_configs:
        logger.error(f"Model {model_name} not found in configuration")
        return False
    
    container_name = model_configs[model_name]["container_name"]
    
    # Get the lock for this specific container
    container_lock = get_container_lock(container_name)
    
    # Acquire the lock to ensure only one thread can stop this container
    with container_lock:
        try:
            # Find the container
            container = get_container(container_name, docker_client)
            
            if not container:
                logger.info(f"Container {container_name} is not running")
                return True
            
            # Stop and remove the container
            logger.info(f"Stopping container {container_name} (ID: {container.id})")
            container.stop(timeout=10)  # Give it 10 seconds to stop gracefully
            container.remove()
            logger.info(f"Container {container_name} stopped and removed")
            return True
            
        except docker.errors.APIError as e:
            logger.error(f"Error stopping container {container_name}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error stopping container {container_name}: {str(e)}")
            return False

def stop_all_containers():
    """Stop all containers in model_configs."""
    logger.info("Stopping all containers...")
    
    for model_name, config in model_configs.items():
        container_name = config["container_name"]
        if is_container_running(model_configs[model_name]["port"], container_name):
            logger.info(f"Stopping container {container_name} for model {model_name}")
            if stop_container(model_name):
                logger.info(f"Successfully stopped container {container_name}")
            else:
                logger.error(f"Failed to stop container {container_name}")
    
    logger.info("All containers stopped")

def update_last_request_time(model_name):
    """Update the last request time for a specific model."""
    with model_lock:
        model_last_request[model_name] = time.time()
        logger.debug(f"Updated last request time for {model_name}")

def process_request(url, data):
    """Stream the request to the specified URL and process the response."""
    try:
        proxies = config.get('proxies', {"http": None, "https": None})
        with requests.post(url, json=data, timeout=REQUEST_TIMEOUT, proxies=proxies, stream=True) as response: 
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    # Process the chunk if needed
                    # For example, you could decode it or transform it
                    processed_chunk = process_chunk(chunk)
                    yield processed_chunk
    except requests.exceptions.Timeout:
        yield json.dumps({"error": "Request to model timed out"}).encode('utf-8')
    except requests.exceptions.RequestException as e:
        yield json.dumps({"error": f"Error forwarding request: {str(e)}"}).encode('utf-8')
    except json.JSONDecodeError:
        yield json.dumps({"error": "Invalid response from model server"}).encode('utf-8')

def process_chunk(chunk):
    """Process a chunk of data."""
    # Example processing: decode the chunk
    try:
        return chunk.decode('utf-8')
    except UnicodeDecodeError:
        return chunk  # Return raw chunk if decoding fails

def load_tokenizer(tokenizer_name):
    """Load a tokenizer from HF and cache it for future use."""
    with tokenizer_cache_lock:
        if tokenizer_name in tokenizer_cache:
            logger.debug(f"Using cached tokenizer for {tokenizer_name}")
            return tokenizer_cache[tokenizer_name]
        
        try:
            logger.info(f"Loading tokenizer {tokenizer_name}")
            if HF_PATH:
                logger.info(f"Loading from cache directory {HF_PATH}")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=HF_PATH, local_files_only=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Cache the tokenizer for future use
            tokenizer_cache[tokenizer_name] = tokenizer
            logger.info(f"Tokenizer {tokenizer_name} loaded and cached successfully")
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer {tokenizer_name}: {str(e)}")
            return None

def handle_request(data, endpoint, service_type):
    """Generalized logic for handling requests."""
    start_time = time.time()
    logger.info(f"Received request from IP: {request.remote_addr} - {endpoint}")
    
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    model_name = data.get('model')
    if not model_name:
        return jsonify({"error": "Model name not specified"}), 400
    
    # Check if the model name is an alternative name and map it to the primary name
    if model_name in model_name_mapping:
        logger.debug(f"Mapping alternative model name '{model_name}' to primary name '{model_name_mapping[model_name]}'")
        model_name = model_name_mapping[model_name]
    
    if model_name not in model_configs:
        return jsonify({"error": f"Model {model_name} not in configuration"}), 404

    # Initialize stats for this model if not already done
    init_model_stats(model_name)

    # Get the container lock to synchronize container operations
    container_name = model_configs[model_name]["container_name"]
    container_lock = get_container_lock(container_name)
     
    # Use the lock to ensure only one thread can check/start the container
    with container_lock:
        # Check if container is running by checking if it's ready
        if not is_container_running(model_configs[model_name]["port"], container_name):
            logger.info(f"Container for {model_name} is not running, checking model availability")
            
            # Check if the model exists in Ollama repository
            if not OLLAMA_PATH:
                logger.error("Ollama path is not configured")
                update_model_stats(model_name, time.time() - start_time, success=False)
                return jsonify({"error": "Ollama path is not configured"}), 400
                
            if not check_model_availability(model_name, is_loading_dashboard=False):
                logger.info(f"Model {model_name} not found in Ollama repository, attempting to download")
                try:
                    # Use ollama pull to download the model
                    logger.info(f"Pulling model {model_name} using ollama pull")
                    result = subprocess.run(['ollama', 'pull', model_name], 
                                         capture_output=True, 
                                         text=True, 
                                         check=True)
                    logger.info(f"Successfully pulled model {model_name}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to pull model {model_name}: {e.stderr}")
                    update_model_stats(model_name, time.time() - start_time, success=False)
                    return jsonify({"error": f"Failed to pull model {model_name}: {e.stderr}"}), 500
                except Exception as e:
                    logger.error(f"Unexpected error pulling model {model_name}: {str(e)}")
                    update_model_stats(model_name, time.time() - start_time, success=False)
                    return jsonify({"error": f"Unexpected error pulling model {model_name}"}), 500
                
                # Check if the model is now available
                if not check_model_availability(model_name, is_loading_dashboard=False):
                    logger.error(f"Model {model_name} still not available after pull attempt")
                    update_model_stats(model_name, time.time() - start_time, success=False)
                    return jsonify({"error": f"Model {model_name} not available after pull attempt"}), 500
                
                logger.info(f"Successfully downloaded model {model_name}")
            
            model_file_path = find_ollama_model_file(OLLAMA_PATH, model_name)
            logger.info(f"Found model {model_name} in Ollama repository at {model_file_path}")
            
            logger.info(f"Starting container for {model_name}")
            is_success = start_container(model_name, service_type, model_file_path)
            if not is_success:
                update_model_stats(model_name, time.time() - start_time, success=False)
                return jsonify({"error": f"Failed to start container for {model_name}"}), 500
            
    # Update last request time
    update_last_request_time(model_name)
    
    url_to_forward = None
    if service_type == 'chat':
        url_to_forward = 'v1/chat/completions'
    elif service_type == 'embedding':
        url_to_forward = 'v1/embeddings'
        
        # Check if this model has HF tokenizer configuration
        model_config = config['models'][model_name]
        if 'tokenizer_from_HF' in model_config and 'model_ctx_size' in model_config:
            model_ctx_size = model_config['model_ctx_size']
            tokenizer_model = model_config['tokenizer_from_HF']
            
            # Get tokenizer from cache or load it if not cached
            tokenizer = load_tokenizer(tokenizer_model)
            
            if tokenizer:
                try:
                    # For embedding models, we need to calculate the prefix length of the model
                    # because the tokenizer will add special tokens to the input text
                    if model_name not in embedding_model_prefix_length:
                        with embedding_model_prefix_length_lock:
                            # Only calculate if not already done
                            with tokenizer_operation_lock:
                                tokens = tokenizer('1', return_tensors="pt", truncation=True, max_length=100, add_special_tokens=True)
                                prefix_length = tokens.input_ids.shape[1] - 1
                                embedding_model_prefix_length[model_name] = prefix_length
                                # Explicitly delete tensor to release memory
                                del tokens
                    
                    safe_max_length = model_ctx_size - embedding_model_prefix_length[model_name]
                    
                    # Process the input data for tokenization

                    if 'input' in data:
                        if isinstance(data['input'], str):
                            # For single string input
                            input_text = data['input']
                            if len(input_text) > 0:
                                with tokenizer_operation_lock:
                                    try:
                                        tokens = tokenizer(input_text, return_tensors="pt", truncation=True, 
                                                       max_length=safe_max_length, add_special_tokens=False)
                                        # Replace the input text with the processed text if it's longer than the max length
                                        if tokens.input_ids.shape[1] >= safe_max_length:
                                            processed_text = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
                                            data['input'] = processed_text
                                        # Explicitly delete tensor
                                        del tokens
                                    except Exception as e:
                                        logger.error(f"Error tokenizing string input: {str(e)}")
                        
                        elif isinstance(data['input'], list):
                            # For list input, process each item separately
                            for i, text in enumerate(data['input']):
                                if isinstance(text, str) and len(text) > 0:
                                    with tokenizer_operation_lock:
                                        try:
                                            tokens = tokenizer(text, return_tensors="pt", truncation=True, 
                                                          max_length=safe_max_length, add_special_tokens=False)
                                            # Replace the input text with the processed text if it's longer than the max length
                                            if tokens.input_ids.shape[1] >= safe_max_length:
                                                processed_text = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
                                                data['input'][i] = processed_text
                                            # Explicitly delete tensor
                                            del tokens
                                        except Exception as e:
                                            logger.error(f"Error tokenizing list item {i}: {str(e)}")
                            
                except Exception as e:
                    logger.error(f"Error using HF tokenizer: {str(e)}")
                    # Continue with default processing if tokenizer fails
                
    # Forward the processed request to the appropriate API endpoint
    if not url_to_forward:
        update_model_stats(model_name, time.time() - start_time, success=False)
        return jsonify({"error": "Invalid service type"}), 400

    # Forward the request
    port = model_configs[model_name]["port"]
    url = f'http://localhost:{port}/{url_to_forward}'
    
    # For embeddings, use non-streaming request to process the response
    if service_type == 'embedding':
        try:
            proxies = config.get('proxies', {"http": None, "https": None})
            
            # Use a direct request instead of streaming for better performance
            response = requests.post(url, json=data, timeout=REQUEST_TIMEOUT, proxies=proxies)
            response.raise_for_status()
            
            # Process the embedding response
            response_json = response.json()
            
            # Round embedding values to match Ollama's precision and format
            if "data" in response_json and isinstance(response_json["data"], list):
                for item in response_json["data"]:
                    if "embedding" in item and isinstance(item["embedding"], list):
                        # Round embedding values to match Ollama's precision (9 decimal places)
                        item["embedding"] = [float(f"{value:.9f}") for value in item["embedding"]]
            
            # Update stats for successful request
            total_time = time.time() - start_time
            update_model_stats(model_name, total_time, success=True)
            
            # Return the modified response
            return jsonify(response_json), response.status_code
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error forwarding request to {model_name}: {str(e)}")
            update_model_stats(model_name, time.time() - start_time, success=False)
            return jsonify({"error": f"Error communicating with model server: {str(e)}"}), 500
        except Exception as e:
            logger.error(f"Error processing embedding response: {str(e)}")
            update_model_stats(model_name, time.time() - start_time, success=False)
            return jsonify({"error": f"Error processing response: {str(e)}"}), 500
    else:
        # For other request types, use streaming response
        try:
            response = Response(process_request(url, data), content_type='application/json')
            # Update stats for successful request
            update_model_stats(model_name, time.time() - start_time, success=True)
            return response
        except Exception as e:
            logger.error(f"Error forwarding request to {model_name}: {str(e)}")
            update_model_stats(model_name, time.time() - start_time, success=False)
            return jsonify({"error": f"Error communicating with model server: {str(e)}"}), 500

@app.route('/v1/chat/completions', methods=['POST'])
def handle_chat_completion():
    """Handle OpenAI chat completions endpoint."""
    return handle_request(request.json, '/v1/chat/completions', 'chat')

@app.route('/api/chat', methods=['POST'])
def handle_api_chat():
    """Handle chat requests to the /api/chat endpoint."""
    return handle_request(request.json, '/api/chat', 'chat')

@app.route('/v1/embeddings', methods=['POST'])
def handle_embeddings():
    """Handle requests to the /v1/embeddings endpoint."""
    return handle_request(request.json, '/v1/embeddings', 'embedding' )

@app.route('/api/embed', methods=['POST'])
def handle_api_embed():
    """Handle requests to the /api/embed endpoint."""
    return handle_request(request.json, '/api/embed', 'embedding' )

def monitor_inactivity():
    """Monitor for inactivity and stop containers that have been idle too long."""
    logger.info("Starting inactivity monitor")
    while True:
        time.sleep(INACTIVITY_CHECK_INTERVAL)  # Check every minute
        current_time = time.time()
        
        with model_lock:
            for model_name, last_time in list(model_last_request.items()):
                if current_time - last_time > INACTIVITY_TIMEOUT:
                    if model_name in model_configs:
                        container_name = model_configs[model_name]["container_name"]
                        if is_container_running(model_configs[model_name]["port"], container_name):
                            logger.info(f"Container {container_name} has been idle for {current_time - last_time:.1f} seconds, stopping")
                            stop_container(model_name)
                            del model_last_request[model_name]

# Ensure wait_for_container_ready is robust
def wait_for_container_ready(model_name, timeout=CONTAINER_STARTUP_TIMEOUT):
    """Wait for the container to be ready, with timeout."""
    container_name = model_configs[model_name]["container_name"]
    if model_name not in model_configs:
        return False
    
    # Give the container a moment to initialize before checking
    logger.info(f"Waiting for initial container startup for {model_name}")
    time.sleep(HEALTH_CHECK_INTERVAL)
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_container_running(model_configs[model_name]["port"], container_name):
            logger.info(f"Container for {model_name} is ready")
            return True
        
        # Log progress during waiting
        elapsed = time.time() - start_time
        if elapsed > HEALTH_CHECK_INTERVAL and elapsed % 10 < HEALTH_CHECK_INTERVAL:  # Log every ~10 seconds
            # Check container logs to see if there are any issues
            try:
                container = get_container(container_name, docker_client, all=False)
                if container:
                    logs = container.logs(tail=10).decode('utf-8', errors='replace')
                    logger.info(f"Waiting for {model_name} to be ready... ({elapsed:.1f}s elapsed)")
                    logger.info(f"Recent container logs: {logs}")
            except Exception as e:
                logger.error(f"Error getting container logs: {str(e)}")
            
        time.sleep(HEALTH_CHECK_INTERVAL)
    
    # If we timed out, get the container logs to help diagnose the issue
    try:
        container_name = model_configs[model_name]["container_name"]
        container = get_container(container_name, docker_client, all=False)
        if container:
            logs = container.logs(tail=50).decode('utf-8', errors='replace')
            logger.error(f"Container logs for {model_name} after timeout: {logs}")
    except Exception as e:
        logger.error(f"Error getting container logs after timeout: {str(e)}")
    
    logger.error(f"Timeout waiting for {model_name} container to be ready")
    return False

def signal_handler(sig, frame):
    """Handle shutdown signals by stopping all containers before exiting."""
    logger.info("Shutdown signal received, stopping all containers...")
    stop_all_containers()
    logger.info("All containers stopped, exiting...")
    sys.exit(0)

def check_model_availability(model_name, is_loading_dashboard=False):
    """Check if a model is available in Ollama repository."""
    return is_ollama_model_available(OLLAMA_PATH, model_name, is_loading_dashboard)

def init_model_stats(model_name):
    """Initialize statistics tracking for a model."""
    if model_name not in model_stats:
        is_downloaded = is_ollama_model_available(OLLAMA_PATH, model_name)
        model_stats[model_name] = {
            "total_requests": 0,
            "successful_requests": 0,
            "response_times": deque(maxlen=1000),  # Keep last 1000 response times
            "request_timestamps": deque(maxlen=1000),  # Keep last 1000 request timestamps
            "request_success": deque(maxlen=1000),  # Track success/failure for each request
            "downloaded": is_downloaded,
            "last_check": time.time()  # Add timestamp of last availability check
        }

def get_stats_for_period(model_name, period_seconds):
    """Calculate statistics for a specific time period."""
    if model_name not in model_stats:
        return 0, 0, 0, 0
    
    stats = model_stats[model_name]
    now = time.time()
    period_start = now - period_seconds
    
    # Get all the requests and their success status within the time period
    period_requests = [(ts, success) for ts, success in zip(stats["request_timestamps"], stats["request_success"]) if ts > period_start]
    
    # Calculate stats directly from the period data
    total_requests = len(period_requests)
    successful_requests = sum(1 for _, success in period_requests if success)
    
    # Get response times for this period for calculating average response time
    period_response_times = [rt for rt, ts in zip(stats["response_times"], stats["request_timestamps"]) if ts > period_start]
    
    # Calculate requests per minute based on first request time
    if total_requests > 0:
        # Find the timestamp of the first request in this period
        first_request_time = min(ts for ts, _ in period_requests) if period_requests else now
        actual_period_duration = now - first_request_time
        # Use at least 1 second to avoid division by zero
        actual_period_duration = max(actual_period_duration, 1)
        requests_per_minute = (total_requests / actual_period_duration) * 60
    else:
        requests_per_minute = 0
    
    avg_response_time = sum(period_response_times) / len(period_response_times) if period_response_times else 0
    
    return total_requests, successful_requests, requests_per_minute, avg_response_time

def update_model_stats(model_name, response_time, success=True):
    """Update statistics for a model after a request."""
    with model_stats_lock:
        if model_name not in model_stats:
            init_model_stats(model_name)
        
        stats = model_stats[model_name]
        stats["total_requests"] += 1
        current_time = time.time()
        
        if success:
            stats["successful_requests"] += 1
            stats["response_times"].append(response_time)
        
        # Record the request timestamp and success status
        stats["request_timestamps"].append(current_time)
        stats["request_success"].append(success)
        
        # Update downloaded status and timestamp
        stats["downloaded"] = is_ollama_model_available(OLLAMA_PATH, model_name)
        stats["last_check"] = current_time

def get_requests_per_minute(model_name):
    """Calculate requests per minute for a model."""
    if model_name not in model_stats:
        return 0
    
    timestamps = model_stats[model_name]["request_timestamps"]
    if not timestamps:
        return 0
    
    # Count requests in the last minute
    now = time.time()
    one_minute_ago = now - 60
    recent_requests = sum(1 for ts in timestamps if ts > one_minute_ago)
    return recent_requests

def get_avg_response_time(model_name):
    """Calculate average response time for a model."""
    if model_name not in model_stats:
        return 0
    
    response_times = model_stats[model_name]["response_times"]
    if not response_times:
        return 0
    
    return sum(response_times) / len(response_times)

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Combined dashboard showing all models in a single table."""
    try:
        # Create a combined list of all models with their status
        all_models = {}
        
        # First, collect all available models
        with model_stats_lock:
            for model_name, model_config in model_configs.items():
                if model_name not in model_stats:
                    init_model_stats(model_name)
                
                # Force check model availability if last check was more than 5 seconds ago
                stats = model_stats[model_name]
                if time.time() - stats["last_check"] > 5:
                    stats["downloaded"] = is_ollama_model_available(OLLAMA_PATH, model_name, is_loading_dashboard=True)
                    stats["last_check"] = time.time()
                
                all_models[model_name] = {
                    "name": model_name,
                    "other_names": model_config.get('other_names', []),
                    "downloaded": stats["downloaded"],
                    "container_name": model_config["container_name"],
                    "is_running": False,
                    "port": "N/A",
                    "device_id": "N/A",
                    "total_requests": 0,
                    "successful_requests": 0,
                    "requests_per_minute": 0,
                    "avg_response_time": 0
                }
        
        # Then check which models are running and update their info
        for model_name, model_config in model_configs.items():
            container_name = model_config["container_name"]
            
            try:
                # Check if container is running with a timeout
                is_running = is_container_running(model_configs[model_name]["port"], container_name)
                
                if is_running and model_name in all_models:
                    # Get statistics with a timeout
                    with model_stats_lock:
                        if model_name not in model_stats:
                            init_model_stats(model_name)
                        
                        stats = model_stats[model_name]
                        
                    all_models[model_name].update({
                        "is_running": True,
                        "port": model_config["port"],
                        "device_id": model_config["device_id"],
                        "total_requests": stats["total_requests"],
                        "successful_requests": stats["successful_requests"],
                        "requests_per_minute": get_requests_per_minute(model_name),
                        "avg_response_time": round(get_avg_response_time(model_name), 3)
                    })
            except Exception as e:
                logger.error(f"Error checking container {container_name}: {str(e)}")
                continue
        
        # Check if JSON format is requested
        if request.args.get('format') == 'json':
            return jsonify({
                "models": list(all_models.values())
            }), 200
        
        # Generate HTML
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Llama.cpp Scheduler Dashboard</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 20px; 
                    background-color: #f9f9f9;
                    color: #333;
                }
                h1, h2 { 
                    color: #2c3e50; 
                    margin-bottom: 20px;
                }
                h1 {
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                table { 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin-top: 20px; 
                    margin-bottom: 40px;
                    font-size: 14px;
                }
                th, td { 
                    padding: 12px 15px; 
                    text-align: left; 
                    border-bottom: 1px solid #e1e1e1; 
                }
                th { 
                    background-color: #3498db; 
                    color: white; 
                    font-weight: bold;
                    position: sticky;
                    top: 0;
                }
                tr:nth-child(even) { background-color: #f7f7f7; }
                tr:hover { background-color: #f1f1f1; }
                .model-name {
                    font-weight: bold;
                    font-size: 15px;
                }
                .other-names { 
                    color: #666; 
                    font-size: 0.9em;
                    font-style: italic;
                    display: block;
                    margin-top: 4px;
                }
                .running {
                    background-color: #e8f5e9;
                }
                .status-true { 
                    color: #2ecc71; 
                    font-weight: bold; 
                }
                .status-false { 
                    color: #e74c3c; 
                }
                .status-running {
                    display: inline-block;
                    padding: 4px 8px;
                    border-radius: 4px;
                    background-color: #2ecc71;
                    color: white;
                    font-weight: bold;
                }
                .status-stopped {
                    display: inline-block;
                    padding: 4px 8px;
                    border-radius: 4px;
                    background-color: #95a5a6;
                    color: white;
                }
                .success-rate { font-weight: bold; }
                .high-rate { color: #2ecc71; }
                .medium-rate { color: #f39c12; }
                .low-rate { color: #e74c3c; }
                .refresh-button { 
                    background-color: #3498db; 
                    color: white; 
                    padding: 10px 20px; 
                    border: none; 
                    border-radius: 4px; 
                    cursor: pointer; 
                    margin-top: 20px;
                    font-size: 16px;
                    transition: background-color 0.3s;
                }
                .refresh-button:hover { 
                    background-color: #2980b9; 
                }
                .section { 
                    margin-bottom: 40px; 
                }
                .last-update { 
                    color: #7f8c8d; 
                    font-size: 0.9em; 
                    margin: 10px 0 20px 0;
                }
                .error-message { 
                    color: #e74c3c; 
                    margin: 10px 0; 
                    padding: 10px;
                    background-color: #fadbd8;
                    border-radius: 4px;
                }
                .time-period-selector {
                    margin: 20px 0;
                }
                .time-period-selector label {
                    margin-right: 15px;
                    font-weight: bold;
                }
                .time-period-buttons {
                    margin-top: 10px;
                    display: flex;
                    gap: 10px;
                }
                .period-button {
                    padding: 8px 15px;
                    border-radius: 4px;
                    border: 1px solid #3498db;
                    background-color: #fff;
                    color: #3498db;
                    cursor: pointer;
                    transition: all 0.3s;
                    font-weight: bold;
                }
                .period-button:hover {
                    background-color: #eaf2f8;
                }
                .period-button.active {
                    background-color: #3498db;
                    color: white;
                }
                .model-row.hidden {
                    display: none;
                }
                .port-gpu-info {
                    white-space: nowrap;
                }
                .port-gpu-label {
                    font-weight: bold;
                    color: #7f8c8d;
                    margin-right: 5px;
                }
            </style>
            <script>
                function refreshPage() {
                    location.reload();
                }
                
                // Auto refresh every 30 mins 
                setTimeout(function() {
                    refreshPage();
                }, 1800000);
                
                // Change time period
                function changeTimePeriod(period) {
                    // Update active button styling
                    document.querySelectorAll('.period-button').forEach(btn => {
                        btn.classList.remove('active');
                    });
                    document.getElementById(period + '-btn').classList.add('active');
                    
                    // Show all rows
                    document.querySelectorAll('.all-time-row, .past-hour-row, .past-day-row').forEach(row => {
                        row.classList.add('hidden');
                    });
                    
                    // Show only selected period rows
                    document.querySelectorAll('.' + period + '-row').forEach(row => {
                        row.classList.remove('hidden');
                    });
                }
                
                // Initialize on load
                window.onload = function() {
                    // Default to all-time view
                    changeTimePeriod('all-time');
                }
            </script>
        </head>
        <body>
            <div class="container">
                <h1>Llama.cpp Scheduler Dashboard</h1>
                <button class="refresh-button" onclick="refreshPage()">Refresh Dashboard</button>
                <p><small>Page auto-refreshes every 30 minutes</small></p>
                <p class="last-update">Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                
                <div class="time-period-selector">
                    <label>Select Time Period:</label>
                    <div class="time-period-buttons">
                        <button id="all-time-btn" class="period-button active" onclick="changeTimePeriod('all-time')">All Time</button>
                        <button id="past-hour-btn" class="period-button" onclick="changeTimePeriod('past-hour')">Past Hour</button>
                        <button id="past-day-btn" class="period-button" onclick="changeTimePeriod('past-day')">Past Day</button>
                    </div>
                </div>
                
                <div class="section">
                    <h2>All Models</h2>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Status</th>
                            <th>Downloaded</th>
                            <th>Total Requests</th>
                            <th>Success Rate</th>
                            <th>Requests/Min</th>
                            <th>Avg Response Time</th>
                        </tr>
        """
        
        # Sort models: running first, then alphabetical
        sorted_models = sorted(all_models.values(), key=lambda x: (not x["is_running"], x["name"]))
        
        for model in sorted_models:
            # Format model name with other names
            model_name_display = f'<span class="model-name">{model["name"]}</span>'
            if model["other_names"]:
                other_names = ", ".join(model["other_names"])
                model_name_display += f'<span class="other-names">({other_names})</span>'
            
            # Format downloaded status
            download_status = "Yes" if model["downloaded"] else "No"
            status_class = "status-true" if model["downloaded"] else "status-false"
            
            # Format running status
            status_display = '<span class="status-running">Running</span>' if model["is_running"] else '<span class="status-stopped">Stopped</span>'
            
            # Format port and GPU info
            if model["is_running"]:
                port_gpu_info = f'<div class="port-gpu-info"><span class="port-gpu-label">Port:</span> {model["port"]}<br><span class="port-gpu-label">GPU:</span> {model["device_id"]}</div>'
            else:
                port_gpu_info = "N/A"
            
            # Get stats for different periods
            all_time = get_stats_for_period(model["name"], float('inf'))
            past_hour = get_stats_for_period(model["name"], 3600)
            past_day = get_stats_for_period(model["name"], 86400)
            
            # Calculate success rates
            all_time_rate = (all_time[1] / all_time[0] * 100) if all_time[0] > 0 else 0
            past_hour_rate = (past_hour[1] / past_hour[0] * 100) if past_hour[0] > 0 else 0
            past_day_rate = (past_day[1] / past_day[0] * 100) if past_day[0] > 0 else 0
            
            # Determine rate classes
            def get_rate_class(rate):
                return "high-rate" if rate > 90 else "medium-rate" if rate > 70 else "low-rate"
            
            all_time_class = get_rate_class(all_time_rate)
            past_hour_class = get_rate_class(past_hour_rate)
            past_day_class = get_rate_class(past_day_rate)
            
            # Row classes
            row_class = "running" if model["is_running"] else ""
            
            # Add one row for each time period (will be shown/hidden via JS)
            html += f"""
                <tr class="model-row all-time-row {row_class}">
                    <td rowspan="1">{model_name_display}</td>
                    <td rowspan="1">{status_display}<br>{port_gpu_info}</td>
                    <td rowspan="1" class="{status_class}">{download_status}</td>
                    <td>{all_time[0]}</td>
                    <td class="success-rate {all_time_class}">{all_time_rate:.1f}%</td>
                    <td>{all_time[2]:.1f}</td>
                    <td>{all_time[3]:.3f}s</td>
                </tr>
                <tr class="model-row past-hour-row hidden {row_class}">
                    <td rowspan="1">{model_name_display}</td>
                    <td rowspan="1">{status_display}<br>{port_gpu_info}</td>
                    <td rowspan="1" class="{status_class}">{download_status}</td>
                    <td>{past_hour[0]}</td>
                    <td class="success-rate {past_hour_class}">{past_hour_rate:.1f}%</td>
                    <td>{past_hour[2]:.1f}</td>
                    <td>{past_hour[3]:.3f}s</td>
                </tr>
                <tr class="model-row past-day-row hidden {row_class}">
                    <td rowspan="1">{model_name_display}</td>
                    <td rowspan="1">{status_display}<br>{port_gpu_info}</td>
                    <td rowspan="1" class="{status_class}">{download_status}</td>
                    <td>{past_day[0]}</td>
                    <td class="success-rate {past_day_class}">{past_day_rate:.1f}%</td>
                    <td>{past_day[2]:.1f}</td>
                    <td>{past_day[3]:.3f}s</td>
                </tr>
            """
        
        html += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        return f"""
        <html>
        <head>
            <title>Error - Llama.cpp Scheduler Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .error-message {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Error Loading Dashboard</h1>
            <p class="error-message">An error occurred while generating the dashboard. Please try refreshing the page.</p>
            <p>Error details: {str(e)}</p>
            <button onclick="location.reload()">Refresh Page</button>
        </body>
        </html>
        """, 500

if __name__ == '__main__':
    logger.info("Starting Llama.cpp scheduler")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Stop all containers when the program starts
    stop_all_containers()
    
    # Start the inactivity monitor in a separate thread
    Thread(target=monitor_inactivity, daemon=True).start()
    
    # Start the Flask app
    debug = config['debug']
    host = config['host']
    port = config['port'] if not debug else config['port'] + 1 # port + 1 for debug
    
    app.run(host=host, port=port, debug=debug)