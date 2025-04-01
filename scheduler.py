import time
from threading import Thread, Lock
from flask import Flask, request, jsonify, Response
import requests
import json
import docker
import signal
import sys
import yaml
from utils import (setup_logger, is_ollama_model_available, find_ollama_model_file, 
                   find_next_available_port, get_best_gpu, process_chunk)
from utils_docker import (start_container, get_container_status, stop_container,
                         wait_for_container_ready, stop_all_containers)

from dashboard_template import (get_dashboard_header, get_dashboard_footer, get_error_page)
from collections import deque
from datetime import datetime
from transformers import AutoTokenizer
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

def process_request(url, data):
    """Stream the request to the specified URL and process the response."""
    proxies = config.get('proxies', {"http": None, "https": None})
    try:
        with requests.post(url, json=data, timeout=REQUEST_TIMEOUT, proxies=proxies, stream=True) as response: 
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    # Process the chunk if needed
                    processed_chunk = process_chunk(chunk)
                    yield processed_chunk
    except requests.exceptions.Timeout:
        yield json.dumps({"error": "Request to model timed out"}).encode('utf-8')
    except requests.exceptions.RequestException as e:
        yield json.dumps({"error": f"Error forwarding request: {str(e)}"}).encode('utf-8')
    except json.JSONDecodeError:
        yield json.dumps({"error": "Invalid response from model server"}).encode('utf-8')

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

def check_model_health(model_name):
    """
    Check if a model's health endpoint is responding correctly.
    
    Args:
        model_name: The name of the model to check
        
    Returns:
        bool: True if health check passes, False otherwise
    """
    if model_name not in model_configs:
        logger.error(f"Model {model_name} not found in configuration")
        return False
        
    port = model_configs[model_name]["port"]
    if not port:
        logger.error(f"No port assigned for model {model_name}")
        return False
        
    health_url = f"http://localhost:{port}/health"
    try:
        proxies = config.get('proxies', {"http": None, "https": None})
        response = requests.get(health_url, timeout=5, proxies=proxies)
        if response.status_code == 200:
            logger.debug(f"Health check for {model_name} successful")
            return True
        else:
            logger.debug(f"Health check for {model_name} failed with status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.debug(f"Health check for {model_name} failed: {str(e)}")
        return False

def get_model_container_ready(model_name, service_type, start_time):
    """
    Prepare a container for the specified model.
    First checks if an existing container is healthy,
    then tries to start an exited container,
    finally creates a new container if needed.
    
    Returns:
        bool: True if container is ready, False if failed
    """
    container_name = model_configs[model_name]["container_name"]
    
    # First, directly check if the model is ready via health check
    # This is the most reliable method and can save time
    if check_model_health(model_name):
        logger.debug(f"Model {model_name} is already running and healthy")
        return True
    
    # If health check failed, check container status to determine next steps
    exists, container_status = get_container_status(container_name, docker_client)
    
    # If container is running but health check failed, it's in a bad state
    if exists and container_status == 'running':
        logger.warning(f"Container {container_name} is running but health check failed, will remove and recreate")
        try:
            container = docker_client.containers.get(container_name)
            container.remove(force=True)
            exists = False  # Reset exists flag to trigger recreation
        except Exception as e:
            logger.error(f"Error removing unhealthy container {container_name}: {str(e)}")
            return False
    
    # Try to start the container if it exists and is in 'exited' state
    if exists and container_status == 'exited':
        logger.info(f"Found exited container {container_name}, attempting to start it")
        try:
            container = docker_client.containers.get(container_name)
            container.start()
            
            # Wait for container to become ready using health check
            max_retries = CONTAINER_STARTUP_TIMEOUT // HEALTH_CHECK_INTERVAL
            retry_count = 0
            
            while retry_count < max_retries:
                if check_model_health(model_name):
                    logger.info(f"Successfully started existing container {container_name}")
                    return True
                time.sleep(HEALTH_CHECK_INTERVAL)
                retry_count += 1
                logger.debug(f"Waiting for model {model_name} to be ready, attempt {retry_count}/{max_retries}")
            
            logger.warning(f"Container {container_name} started but health check failed, will remove and recreate")
            # Remove the container since it failed to become ready
            container.remove(force=True)
            exists = False  # Reset exists flag to trigger recreation
        except Exception as e:
            logger.warning(f"Failed to start existing container {container_name}: {str(e)}, will remove and recreate")
            # Try to remove the container so we can create a new one
            try:
                container = docker_client.containers.get(container_name)
                container.remove(force=True)
            except Exception as remove_error:
                logger.error(f"Error removing container after failed start: {str(remove_error)}")
            exists = False  # Reset exists flag to trigger recreation
    
    # If container doesn't exist or needs recreation, check model availability and create new container
    if not exists:
        if not OLLAMA_PATH:
            logger.error("Ollama path is not configured")
            update_model_stats(model_name, time.time() - start_time, success=False)
            return False
            
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
                return False
            except Exception as e:
                logger.error(f"Unexpected error pulling model {model_name}: {str(e)}")
                update_model_stats(model_name, time.time() - start_time, success=False)
                return False
            
            # Check if the model is now available
            if not check_model_availability(model_name, is_loading_dashboard=False):
                logger.error(f"Model {model_name} still not available after pull attempt")
                update_model_stats(model_name, time.time() - start_time, success=False)
                return False
            
            logger.info(f"Successfully downloaded model {model_name}")
        
        model_file_path = find_ollama_model_file(OLLAMA_PATH, model_name)
        logger.info(f"Found model {model_name} in Ollama repository at {model_file_path}")
        
        logger.info(f"Starting container for {model_name}")
        is_success = start_model_container(model_name, service_type, model_file_path)
        return is_success
    
    # If we reach here, something unexpected happened
    logger.error(f"Failed to prepare container for {model_name}, state: exists={exists}, status={container_status}")
    return False

def process_embedding_input(data, model_name, model_ctx_size, tokenizer_model):
    """
    Process embedding input with the appropriate tokenizer and truncate if necessary.
    
    Args:
        data: The request data containing input text
        model_name: Name of the model
        model_ctx_size: Context size of the model
        tokenizer_model: HuggingFace tokenizer model name
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Get tokenizer from cache or load it if not cached
    tokenizer = load_tokenizer(tokenizer_model)
    if not tokenizer:
        return False
        
    try:
        # Calculate prefix length if not already done
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
        return True
                
    except Exception as e:
        logger.error(f"Error using HF tokenizer: {str(e)}")
        return False

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
        # Prepare the container for this model
        if not get_model_container_ready(model_name, service_type, start_time):
            return jsonify({"error": f"Failed to prepare container for {model_name}"}), 500
            
    # Update last request time
    update_last_request_time(model_name)
    
    # Determine endpoint to forward request to
    url_to_forward = 'v1/chat/completions' if service_type == 'chat' else 'v1/embeddings' if service_type == 'embedding' else None
    
    # Handle tokenization for embedding requests if needed
    if service_type == 'embedding':
        # Check if this model has HF tokenizer configuration
        model_config = config['models'][model_name]
        if 'tokenizer_from_HF' in model_config and 'model_ctx_size' in model_config:
            # Process embedding input with tokenizer
            process_embedding_input(
                data, 
                model_name, 
                model_config['model_ctx_size'], 
                model_config['tokenizer_from_HF']
            )
    
    # If no valid endpoint, return error        
    if not url_to_forward:
        update_model_stats(model_name, time.time() - start_time, success=False)
        return jsonify({"error": "Invalid service type"}), 400

    # Prepare for request
    port = model_configs[model_name]["port"]
    url = f'http://localhost:{port}/{url_to_forward}'
    
    try:
        # Use different handling for embedding vs. chat requests
        if service_type == 'embedding':
            # Direct request (non-streaming) for embeddings
            proxies = config.get('proxies', {"http": None, "https": None})
            response = requests.post(url, json=data, timeout=REQUEST_TIMEOUT, proxies=proxies)
            response.raise_for_status()
            
            # Process the embedding response
            response_json = response.json()
            
            # Round embedding values to match Ollama's precision
            if "data" in response_json and isinstance(response_json["data"], list):
                for item in response_json["data"]:
                    if "embedding" in item and isinstance(item["embedding"], list):
                        item["embedding"] = [float(f"{value:.9f}") for value in item["embedding"]]
            
            result = jsonify(response_json), response.status_code
        else:
            # Streaming response for chat completions
            result = Response(process_request(url, data), content_type='application/json')
        
        # Update stats for successful request
        update_model_stats(model_name, time.time() - start_time, success=True)
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error forwarding request to {model_name}: {str(e)}")
        update_model_stats(model_name, time.time() - start_time, success=False)
        return jsonify({"error": f"Error communicating with model server: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error processing response: {str(e)}")
        update_model_stats(model_name, time.time() - start_time, success=False)
        return jsonify({"error": f"Error processing response: {str(e)}"}), 500


def start_model_container(model_name, service_type, model_file_path=None):
    """Start a Docker container for a specific model."""
    container_name = model_configs[model_name]["container_name"]

    # Get the best GPU to use if not already assigned
    if model_configs[model_name]["device_id"] is None:
        model_configs[model_name]["device_id"] = get_best_gpu(FORCE_GPU_ID)
    
    # Find next available port if not already assigned
    if model_configs[model_name]["port"] is None:
        model_configs[model_name]["port"] = find_next_available_port(start_port=8090)
        
    # Start the container using the utility function
    result = start_container(
        model_name, 
        model_configs, 
        docker_client, 
        DOCKER_IMAGE, 
        service_type, 
        model_file_path, 
        OLLAMA_PATH, 
        HF_MIRROR
    )
    
    if result:
        # Wait for container to become ready using health check
        max_retries = CONTAINER_STARTUP_TIMEOUT // HEALTH_CHECK_INTERVAL
        retry_count = 0
        is_healthy = False
            
        while retry_count < max_retries:
            if check_model_health(model_name):
                is_healthy = True
                break
            time.sleep(HEALTH_CHECK_INTERVAL)
            retry_count += 1
            logger.debug(f"Waiting for model {model_name} to be ready, attempt {retry_count}/{max_retries}")
            
        if is_healthy:
            logger.info(f"Container {container_name} is running and healthy")
            return True
        else:
            logger.error(f"Container {container_name} failed to become healthy")
            # Try to remove the unhealthy container
            try:
                container = docker_client.containers.get(container_name)
                container.remove(force=True)
            except Exception as e:
                logger.error(f"Error removing unhealthy container: {str(e)}")
            return False
                
        return result

def monitor_inactivity():
    """Monitor for inactivity and stop containers that have been idle too long."""
    logger.info("Starting inactivity monitor")
    while True:
        time.sleep(INACTIVITY_CHECK_INTERVAL)  # Check every minute
        current_time = time.time()
        
        # Create a list of models to stop without holding the model_lock
        models_to_stop = []
        
        with model_lock:
            for model_name, last_time in list(model_last_request.items()):
                if current_time - last_time > INACTIVITY_TIMEOUT:
                    if model_name in model_configs:
                        models_to_stop.append(model_name)
        
        # Now stop the containers without holding the model_lock
        for model_name in models_to_stop:
            container_name = model_configs[model_name]["container_name"]
            # Check if container exists in any state
            exists, status = get_container_status(container_name, docker_client)
            if exists:
                logger.info(f"Container {container_name} has been idle for {current_time - model_last_request.get(model_name, 0):.1f} seconds, stopping (current status: {status})")
                
                # Get the lock for this specific container
                container_lock = get_container_lock(container_name)
                
                # Acquire the lock to ensure only one thread can stop this container
                with container_lock:
                    logger.info(f"Stopping container {container_name} for model {model_name}")
                    # Use the utility function to stop and remove the container
                    stop_result = stop_container(container_name, docker_client, remove=True)
                    if not stop_result:
                        logger.error(f"Failed to stop container {container_name}")
                
                # Now remove from last request tracking
                with model_lock:
                    if model_name in model_last_request:
                        del model_last_request[model_name]

def signal_handler(sig, frame):
    """Handle shutdown signals by stopping all containers before exiting."""
    logger.info("Shutdown signal received, stopping all containers...")
    stop_all_containers(model_configs, docker_client)
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

def get_stats_for_period(model_name, period_seconds, stats_data):
    """Calculate statistics for a specific time period using provided stats data."""
    # Use the stats data passed in rather than acquiring locks again
    if not stats_data:
        return 0, 0, 0, 0
    
    now = time.time()
    period_start = now - period_seconds
    
    # Get all the requests and their success status within the time period
    period_requests = [(ts, success) for ts, success in zip(stats_data["request_timestamps"], stats_data["request_success"]) if ts > period_start]
    
    # Calculate stats directly from the period data
    total_requests = stats_data["total_requests"]
    successful_requests = stats_data["successful_requests"]

    requests_in_period = len(period_requests)
    
    # Get response times for this period for calculating average response time
    period_response_times = [rt for rt, ts in zip(stats_data["response_times"], stats_data["request_timestamps"]) if ts > period_start]
    
    # Calculate requests per minute based on first request time
    if requests_in_period > 0:
        # Find the timestamp of the first request in this period
        first_request_time = min(ts for ts, _ in period_requests) if period_requests else now
        actual_period_duration = now - first_request_time
        # Use at least 1 second to avoid division by zero
        actual_period_duration = max(actual_period_duration, 1)
        requests_per_minute = (requests_in_period / actual_period_duration) * 60
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

def update_last_request_time(model_name):
    """Update the last request time for a specific model."""
    with model_lock:
        model_last_request[model_name] = time.time()
        logger.debug(f"Updated last request time for {model_name}")

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Combined dashboard showing all models in a single table."""
    try:
        # Create a combined list of all models with their status
        all_models = {}
        
        # First, collect all available models
        model_names = list(model_configs.keys())
        
        # Get current time for inactivity calculations
        current_time = time.time()
        
        # Get all model stats in one lock acquisition
        with model_stats_lock:
            for model_name in model_names:
                if model_name not in model_stats:
                    init_model_stats(model_name)
                
                # Get stats object without checking availability yet
                stats = model_stats[model_name]
                
                all_models[model_name] = {
                    "name": model_name,
                    "other_names": model_configs[model_name].get('other_names', []),
                    "downloaded": stats["downloaded"],
                    "container_name": model_configs[model_name]["container_name"],
                    "is_running": False,
                    "port": "N/A",
                    "device_id": "N/A",
                    "total_requests": stats["total_requests"],
                    "successful_requests": stats["successful_requests"],
                    "response_times": list(stats["response_times"]),
                    "request_timestamps": list(stats["request_timestamps"]),
                    "request_success": list(stats["request_success"]),
                    "last_check": stats["last_check"],
                    "last_request_time": model_last_request.get(model_name, 0)
                }
        
        # Now update availability info separately to avoid lock nesting
        for model_name, model_data in all_models.items():
            # Force check model availability if last check was more than 5 seconds ago
            if time.time() - model_data["last_check"] > 5:
                is_available = is_ollama_model_available(OLLAMA_PATH, model_name, is_loading_dashboard=True)
                
                # Update the stats with the new availability info
                with model_stats_lock:
                    if model_name in model_stats:
                        model_stats[model_name]["downloaded"] = is_available
                        model_stats[model_name]["last_check"] = time.time()
                
                model_data["downloaded"] = is_available
        
        # Then check which models are running and update their info
        for model_name, model_config in model_configs.items():
            container_name = model_config["container_name"]
            
            try:
                # Check container status
                exists, container_status = get_container_status(container_name, docker_client)
                
                if exists and model_name in all_models:
                    # Get statistics with a timeout
                    with model_stats_lock:
                        if model_name not in model_stats:
                            init_model_stats(model_name)
                        
                        stats = model_stats[model_name]
                    
                    # Update is_running based on container status
                    is_running = container_status == 'running'
                    
                    all_models[model_name].update({
                        "is_running": is_running,
                        "container_status": container_status,
                        "port": model_config["port"] if is_running else "N/A",
                        "device_id": model_config["device_id"] if is_running else "N/A",
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
        
        # Use template for HTML header
        html = get_dashboard_header(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
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
            download_status_class = "status-true" if model["downloaded"] else "status-false"
            
            # Format running status
            exists, container_status = get_container_status(model["container_name"], docker_client)
            if exists:
                if container_status == 'running':
                    # Calculate time until unload
                    with model_lock:
                        last_request_time = model_last_request.get(model["name"], current_time)
                    
                    time_since_last_request = current_time - last_request_time
                    time_until_unload = max(0, INACTIVITY_TIMEOUT - time_since_last_request)
                    
                    # Format time until unload
                    if time_until_unload > 0:
                        minutes, seconds = divmod(int(time_until_unload), 60)
                        hours, minutes = divmod(minutes, 60)
                        
                        if hours > 0:
                            time_format = f"{hours}h {minutes}m"
                        else:
                            time_format = f"{minutes}m {seconds}s"
                    else:
                        time_format = "unloading soon..."
                    
                    # Add the unload timer on a new line between Running and port info
                    status_display = f'<span class="status-running">Running</span><br><span class="unload-timer">{time_format}</span>'
                    status_class = "status-running"
                else:
                    status_class = f"status-{container_status}" if container_status in ['exited', 'created', 'restarting', 'paused'] else "status-stopped"
                    status_display = f'<span class="{status_class}">{container_status.capitalize()}</span>'
            else:
                status_display = '<span class="status-stopped">Stopped</span>'
                status_class = "status-stopped"
            
            # Update is_running for statistics
            model["is_running"] = exists and container_status == 'running'
            
            # Format port and GPU info
            if model["is_running"]:
                port_gpu_info = f'<div class="port-gpu-info"><span class="port-gpu-label">Port:</span> {model["port"]}<br><span class="port-gpu-label">GPU:</span> {model["device_id"]}</div>'
            else:
                port_gpu_info = "N/A"
            
            # Get stats for different periods
            all_time = get_stats_for_period(model["name"], float('inf'), model_stats[model["name"]])
            past_hour = get_stats_for_period(model["name"], 3600, model_stats[model["name"]])
            past_day = get_stats_for_period(model["name"], 86400, model_stats[model["name"]])
            
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
                    <td rowspan="1" class="{download_status_class}">{download_status}</td>
                    <td>{all_time[0]}</td>
                    <td class="success-rate {all_time_class}">{all_time_rate:.1f}%</td>
                    <td>{all_time[2]:.1f}</td>
                    <td>{all_time[3]:.3f}s</td>
                </tr>
                <tr class="model-row past-hour-row hidden {row_class}">
                    <td rowspan="1">{model_name_display}</td>
                    <td rowspan="1">{status_display}<br>{port_gpu_info}</td>
                    <td rowspan="1" class="{download_status_class}">{download_status}</td>
                    <td>{past_hour[0]}</td>
                    <td class="success-rate {past_hour_class}">{past_hour_rate:.1f}%</td>
                    <td>{past_hour[2]:.1f}</td>
                    <td>{past_hour[3]:.3f}s</td>
                </tr>
                <tr class="model-row past-day-row hidden {row_class}">
                    <td rowspan="1">{model_name_display}</td>
                    <td rowspan="1">{status_display}<br>{port_gpu_info}</td>
                    <td rowspan="1" class="{download_status_class}">{download_status}</td>
                    <td>{past_day[0]}</td>
                    <td class="success-rate {past_day_class}">{past_day_rate:.1f}%</td>
                    <td>{past_day[2]:.1f}</td>
                    <td>{past_day[3]:.3f}s</td>
                </tr>
            """
        
        # Use template for HTML footer
        html += get_dashboard_footer()
        
        return html
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        return get_error_page(str(e)), 500


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

if __name__ == '__main__':
    logger.info("Starting Llama.cpp scheduler")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Stop and remove all containers when the program starts to ensure a clean state
    logger.info("Stopping and removing all existing containers")
    stop_all_containers(model_configs, docker_client)
    
    # Start the inactivity monitor in a separate thread
    Thread(target=monitor_inactivity, daemon=True).start()
    
    # Start the Flask app
    if config['debug']:
        port = find_next_available_port(start_port=config['port'] + 1) # + 1  to avoid conflict with the running port
    else:
        port = find_next_available_port(start_port=config['port'])
    
    app.run(host=config['host'], port=port, debug=config['debug'])