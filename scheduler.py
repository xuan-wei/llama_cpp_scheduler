import time
from threading import Thread, Lock
from flask import Flask, request, jsonify
import requests
import json
import docker
import signal
import sys
import yaml
from utils import (get_best_gpu, find_next_available_port, ensure_models_exist, 
                   is_container_running, get_container, setup_logger)
from pathlib import Path

logger = setup_logger('scheduler')

app = Flask(__name__)

# Docker client
docker_client = docker.from_env()  # Run as root

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update global variables from config
MODEL_BASE_PATH = config['model_base_path']
DOCKER_IMAGE = config['docker_image']
INACTIVITY_TIMEOUT = config['inactivity_timeout']
INACTIVITY_CHECK_INTERVAL = config['inactivity_check_interval']
CONTAINER_STARTUP_TIMEOUT = config['container_startup_timeout']
HEALTH_CHECK_INTERVAL = config['health_check_interval']
MODEL_INIT_DELAY = config['model_init_delay']
FORCE_GPU_ID = config['force_gpu_id']

# Initialize model configurations from config file
model_configs = {}
for model_name, model_config in config['models'].items():
    container_name = model_config["container_name"]
    container_name = container_name.replace("llama.cpp_", "llama.cpp_scheduler_")
    model_configs[model_name] = {
        **config['common_config'],
        **model_config,
        "container_name": container_name
    }

# Dictionary to track the last request time for each model
model_last_request = {}
model_lock = Lock()  # Lock for thread-safe operations on the model_last_request dict

# Container operation locks to prevent multiple simultaneous start operations
container_locks = {}
container_locks_lock = Lock()  # Meta-lock to protect the container_locks dictionary

# Initialize last request time for all models
for model_name in model_configs:
    model_last_request[model_name] = time.time()

def get_container_lock(container_name):
    """Get or create a lock for a specific container."""
    with container_locks_lock:
        if container_name not in container_locks:
            container_locks[container_name] = Lock()
        return container_locks[container_name]

def start_container(model_name):
    """Start a Docker container with the specified parameters using Docker SDK."""
    if model_name not in model_configs:
        logger.error(f"Model {model_name} not found in configuration")
        return False
    
    config = model_configs[model_name]
    container_name = config["container_name"]
    
    try:
        # Check if container is already running
        if is_container_running(container_name, docker_client):
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
            "CUDA_VISIBLE_DEVICES": str(config["device_id"])
        }
        
        # Prepare port mapping
        ports = {f"{config['port']}/tcp": config['port']}
                
        # Prepare command
        command = [
            "-m", f"/models/{model_name}.gguf",
            "-c", str(config["ctx_size"]),
            "--n-gpu-layers", str(config["n_gpu_layers"]),
            "--host", "0.0.0.0",
            "--port", str(config["port"]),
            "--threads", str(config["threads"]),
            "--parallel", str(config["parallel"]),
            "--cont-batching",
            "--flash-attn",

        ]

        volumes = {
            MODEL_BASE_PATH: {'bind': '/models', 'mode': 'ro'}
        }
        
        logger.info(f"Starting container {container_name} with command: {' '.join(command)} --device-id {config['device_id']}")
        
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
        if is_container_running(container_name, docker_client):
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

def forward_request(url, data):
    """Forward the request to the specified URL."""
    try:
        proxies = config.get('proxies', {"http": None, "https": None})
        response = requests.post(url, json=data, timeout=120, proxies=proxies)  # 2-minute timeout
        return jsonify(response.json()), response.status_code
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request to model timed out"}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error forwarding request: {str(e)}"}), 500
    except json.JSONDecodeError:
        # Handle case where response is not valid JSON
        return jsonify({"error": "Invalid response from model server"}), 502

@app.route('/v1/chat/completions', methods=['POST'])
def handle_chat_completion():
    """Handle OpenAI chat completions endpoint."""
    logger.info(f"Received request from IP: {request.remote_addr} - /v1/chat/completions")
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    model_name = data.get('model')
    if not model_name:
        return jsonify({"error": "Model name not specified"}), 400
    
    if model_name not in model_configs:
        return jsonify({"error": f"Model {model_name} not found"}), 404

    # Check if model file exists
    model_path = Path(MODEL_BASE_PATH) / f"{model_name}.gguf"
    if not model_path.exists():
        # Only download if model doesn't exist
        model_config = config['models'][model_name]
        if not ensure_models_exist({model_name: model_config}, MODEL_BASE_PATH):
            return jsonify({"error": f"Failed to download model {model_name}"}), 500

    # Get the container lock to synchronize container operations
    container_name = model_configs[model_name]["container_name"]
    container_lock = get_container_lock(container_name)
     
    # Use the lock to ensure only one thread can check/start the container
    with container_lock:
        # Check if container is running by checking if it's ready
        if not is_container_running(container_name, docker_client):
            logger.info(f"Container for {model_name} is not running, attempting to start it")
            is_success = start_container(model_name)
            if not is_success:
                return jsonify({"error": f"Failed to start container for {model_name}"}), 500
            
    # Update last request time
    update_last_request_time(model_name)
    
    # Forward the request
    port = model_configs[model_name]["port"]
    url = f'http://localhost:{port}/v1/chat/completions'
    try:
        return forward_request(url, data)
    except Exception as e:
        logger.error(f"Error forwarding request to {model_name}: {str(e)}")
        return jsonify({"error": f"Error communicating with model server: {str(e)}"}), 500

@app.route('/admin/start', methods=['POST'])
def admin_start_container():
    """Admin endpoint to manually start a container."""
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    model_name = data.get('model')
    if not model_name:
        return jsonify({"error": "Model name not specified"}), 400
    
    if model_name not in model_configs:
        return jsonify({"error": f"Model {model_name} not found"}), 404
    
    container_name = model_configs[model_name]["container_name"]
    
    # Get the container lock
    container_lock = get_container_lock(container_name)
    
    # Use the lock to ensure only one thread can start the container
    with container_lock:
        # Stop the container first if it's already running
        if is_container_running(container_name, docker_client):
            logger.info(f"Container {container_name} is already running, stopping it first to apply new settings")
            stop_container(model_name)
        
        is_success = start_container(model_name)
        if not is_success:
            return jsonify({"error": f"Failed to start container for {model_name}"}), 500
        
        # Wait for container to be ready
        is_ready = wait_for_container_ready(model_name)
        if not is_ready:
            return jsonify({"error": f"Container for {model_name} failed to start properly"}), 500
    
    return jsonify({
        "message": f"Container for {model_name} started successfully",
        "port": model_configs[model_name]["port"],
        "device_id": model_configs[model_name]["device_id"]
    }), 200

@app.route('/admin/stop', methods=['POST'])
def admin_stop_container():
    """Admin endpoint to manually stop a container."""
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    model_name = data.get('model')
    if not model_name:
        return jsonify({"error": "Model name not specified"}), 400
    
    if model_name not in model_configs:
        return jsonify({"error": f"Model {model_name} not found"}), 404
    
    is_success = stop_container(model_name)
    if not is_success:
        return jsonify({"error": f"Failed to stop container for {model_name}"}), 500
    
    # Remove from last request time tracking
    with model_lock:
        if model_name in model_last_request:
            del model_last_request[model_name]
    
    return jsonify({"message": f"Container for {model_name} stopped successfully"}), 200

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
                        if is_container_running(container_name, docker_client):
                            logger.info(f"Container {container_name} has been idle for {current_time - last_time:.1f} seconds, stopping")
                            stop_container(model_name)
                            del model_last_request[model_name]

# Ensure wait_for_container_ready is robust
def wait_for_container_ready(model_name, timeout=CONTAINER_STARTUP_TIMEOUT):
    """Wait for the container to be ready, with timeout."""
    if model_name not in model_configs:
        return False
    
    # Give the container a moment to initialize before checking
    logger.info(f"Waiting {MODEL_INIT_DELAY} seconds for initial container startup for {model_name}")
    time.sleep(MODEL_INIT_DELAY)
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_container_running(model_name, docker_client):
            logger.info(f"Container for {model_name} is ready")
            return True
        
        # Log progress during waiting
        elapsed = time.time() - start_time
        if elapsed > 5 and elapsed % 10 < HEALTH_CHECK_INTERVAL:  # Log every ~10 seconds
            # Check container logs to see if there are any issues
            try:
                container_name = model_configs[model_name]["container_name"]
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
    app.run(host='0.0.0.0', port=5000, debug=False)