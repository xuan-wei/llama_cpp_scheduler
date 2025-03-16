import subprocess
import time
import logging
from threading import Thread, Lock
from flask import Flask, request, jsonify, Response
import requests
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scheduler')

app = Flask(__name__)

# Configuration
INACTIVITY_TIMEOUT = 60  # seconds
INACTIVITY_CHECK_INTERVAL = 30  # seconds
CONTAINER_STARTUP_TIMEOUT = 30  # seconds
HEALTH_CHECK_INTERVAL = 1  # seconds

PROXIES_SETTINGS = {
    "http": None,
    "https": None
}

# Dictionary to track the last request time for each model
model_last_request = {}
model_lock = Lock()  # Lock for thread-safe operations on the model_last_request dict

# Container operation locks to prevent multiple simultaneous start operations
container_locks = {}
container_locks_lock = Lock()  # Meta-lock to protect the container_locks dictionary

# Mapping of model names to container names
model_to_container = {
    "qwen2.5-14b": "llama.cpp_qwen2.5-14b",
    "nanbeige-16b": "llama.cpp_nanbeige-16b",
    "phi4-14b": "llama.cpp_phi4-14b",
    "qwen2.5-7b": "llama.cpp_qwen2.5-7b",
    "qwen2.5-coder-7b": "llama.cpp_qwen2.5-coder-7b",
    "deepseek-coder-6.7b": "llama.cpp_deepseek-coder-6.7b",
    "glm4": "llama.cpp_glm4",
    "llama3.1-8b": "llama.cpp_llama3.1-8b"
}

# Mapping model names to ports
model_ports = {
    "qwen2.5-14b": 8091,
    "nanbeige-16b": 8092,
    "phi4-14b": 8093,
    "qwen2.5-7b": 8094,
    "qwen2.5-coder-7b": 8095,
    "deepseek-coder-6.7b": 8096,
    "glm4": 8097,
    "llama3.1-8b": 8098
}

def get_container_lock(container_name):
    """Get or create a lock for a specific container."""
    with container_locks_lock:
        if container_name not in container_locks:
            container_locks[container_name] = Lock()
        return container_locks[container_name]

def start_container(container_name):
    """Start the Docker container if it's not running."""
    
    try:
        result = subprocess.run(
            ["sudo", "docker", "ps", "-q", "-f", f"name={container_name}"],
            capture_output=True,
            text=True,
            check=True
        )
        if not result.stdout.strip():
            logger.info(f"Starting container: {container_name}")
            subprocess.run(["sudo", "docker", "compose", "up", "-d", container_name], check=True)
            return True
        else:
            logger.info(f"Container {container_name} is already running")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting container {container_name}: {str(e)}")
        return False

def stop_container(container_name):
    """Stop the Docker container."""
    # Get the lock for this specific container
    container_lock = get_container_lock(container_name)
    
    # Acquire the lock to ensure only one thread can stop this container
    with container_lock:
        try:
            logger.info(f"Stopping container: {container_name}")
            subprocess.run(["sudo", "docker", "compose", "down", container_name], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error stopping container {container_name}: {str(e)}")
            return False

def is_container_ready(model_name):
    port = model_ports.get(model_name)
    """Check if the container is ready to accept requests."""
    try:
        response = requests.get(f"http://localhost:{port}/v1/models", timeout=2, proxies=PROXIES_SETTINGS)
        return response.status_code == 200
    except requests.RequestException:
        return False

def wait_for_container_ready(model_name, timeout=CONTAINER_STARTUP_TIMEOUT):
    """Wait for the container to be ready, with timeout."""
    port = model_ports.get(model_name)
    if not port:
        return False
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_container_ready(model_name):
            logger.info(f"Container for {model_name} is ready")
            return True
        time.sleep(HEALTH_CHECK_INTERVAL)
    
    logger.error(f"Timeout waiting for {model_name} container to be ready")
    return False

def update_last_request_time(model_name):
    """Update the last request time for a specific model."""
    with model_lock:
        model_last_request[model_name] = time.time()
        logger.debug(f"Updated last request time for {model_name}")

def forward_request(url, data):
    """Forward the request to the specified URL."""
    try:
        response = requests.post(url, json=data, timeout=120, proxies=PROXIES_SETTINGS)  # 2-minute timeout
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
    
    container_name = model_to_container.get(model_name)
    if not container_name:
        return jsonify({"error": f"Model {model_name} not found"}), 404

    # Get the container lock to synchronize container operations
    container_lock = get_container_lock(container_name)
    
    # Use the lock to ensure only one thread can check/start the container
    with container_lock:
        # Check if container is running by checking if it's ready
        if not is_container_ready(model_name):
            logger.info(f"Container for {model_name} is not ready, attempting to start it")
            is_success = start_container(container_name)
            if not is_success:
                return jsonify({"error": f"Failed to start container for {model_name}"}), 500
            
            # Wait for container to be ready
            is_ready =  wait_for_container_ready(model_name)
            if not is_ready:
                return jsonify({"error": f"Container for {model_name} failed to start properly"}), 500
    
    # Update last request time
    update_last_request_time(model_name)
    
    # Forward the request
    url = f'http://localhost:{model_ports[model_name]}/v1/chat/completions'
    return forward_request(url, data)

def monitor_inactivity():
    """Monitor for inactivity and stop containers that have been idle too long."""
    logger.info("Starting inactivity monitor")
    while True:
        time.sleep(INACTIVITY_CHECK_INTERVAL)  # Check every minute
        current_time = time.time()
        
        with model_lock:
            for model_name, last_time in list(model_last_request.items()):
                if current_time - last_time > INACTIVITY_TIMEOUT:
                    container_name = model_to_container.get(model_name)
                    if is_container_ready(model_name):
                        logger.info(f"Container {container_name} has been idle for {current_time - last_time:.1f} seconds, stopping")
                        stop_container(container_name)
                        del model_last_request[model_name]

if __name__ == '__main__':
    logger.info("Starting Llama.cpp scheduler")
    # Start the inactivity monitor in a separate thread
    Thread(target=monitor_inactivity, daemon=True).start()
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False) 