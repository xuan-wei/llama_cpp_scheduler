import docker
from utils import setup_logger
import os

logger = setup_logger('docker_utils')

def is_container_running(container_name, docker_client):
    """Check if a container is running using Docker SDK."""
    try:
        # Only get running containers
        containers = docker_client.containers.list(filters={"name": container_name}, all=False)
        return len(containers) > 0
    except docker.errors.APIError as e:
        logger.error(f"Error checking if container {container_name} is running: {str(e)}")
        return False

def get_container(container_name, docker_client, all=True):
    """Get a container by name, including stopped containers if all=True."""
    try:
        containers = docker_client.containers.list(filters={"name": container_name}, all=all)
        return containers[0] if containers else None
    except docker.errors.APIError as e:
        logger.error(f"Error getting container {container_name}: {str(e)}")
        return None

def get_container_status(container_name, docker_client):
    """
    Get detailed status of a container by name.
    Returns a tuple: (exists, status_string)
    Where status_string is one of: 'running', 'exited', 'created', 'paused', 'restarting', etc.
    """
    try:
        container = get_container(container_name, docker_client, all=True)
        if container:
            return True, container.status
        return False, None
    except docker.errors.APIError as e:
        logger.error(f"Error getting status for container {container_name}: {str(e)}")
        return False, None

def stop_container(container_name, docker_client, remove=False):
    """
    Stop a container if it exists and is running or in another active state.
    Optionally remove the container after stopping.
    Returns True if operation was successful or if container doesn't exist.
    """
    try:
        container = get_container(container_name, docker_client, all=True)
        
        if not container:
            logger.info(f"Container {container_name} does not exist")
            return True
            
        exists, status = get_container_status(container_name, docker_client)
        
        if exists:
            if status == 'running' or status in ['restarting', 'paused']:
                logger.info(f"Stopping container {container_name} (current status: {status})")
                container.stop(timeout=30)
                logger.info(f"Container {container_name} stopped")
            else:
                logger.info(f"Container {container_name} is already stopped (status: {status})")
                
            if remove:
                logger.info(f"Removing container {container_name}")
                container.remove(force=True)
                logger.info(f"Container {container_name} removed")
                
        return True
    except docker.errors.APIError as e:
        logger.error(f"Error stopping/removing container {container_name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error with container {container_name}: {str(e)}")
        return False

def remove_container_if_exists(container_name, docker_client):
    """
    Remove a container if it exists, regardless of its status.
    Returns True if successful or if container doesn't exist.
    """
    try:
        container = get_container(container_name, docker_client, all=True)
        
        if not container:
            logger.info(f"Container {container_name} does not exist, nothing to remove")
            return True
            
        # Force remove handles any container state
        logger.info(f"Removing container {container_name} (status: {container.status})")
        container.remove(force=True)
        logger.info(f"Container {container_name} successfully removed")
        return True
    except docker.errors.APIError as e:
        logger.error(f"Error removing container {container_name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error removing container {container_name}: {str(e)}")
        return False

def start_container(model_name, model_configs, docker_client, docker_image, 
                   service_type, model_file_path=None, ollama_path='', hf_mirror=''):
    """Start a Docker container with the specified parameters using Docker SDK."""
    if model_name not in model_configs:
        logger.error(f"Model {model_name} not found in configuration")
        return False
    
    config = model_configs[model_name]
    container_name = config["container_name"]
    
    # Get the ollama model name for logging
    ollama_model_name = config.get("ollama_name", model_name)
    
    try:
        # Check if container is already running
        if is_container_running(container_name, docker_client):
            logger.info(f"Container {container_name} is already running")
            return True
        
        # Check container status
        exists, status = get_container_status(container_name, docker_client)
        if exists:
            logger.info(f"Found existing container {container_name} in state: {status}")
            # Remove the container regardless of its state
            if not remove_container_if_exists(container_name, docker_client):
                logger.error(f"Failed to remove existing container {container_name}")
                return False
            logger.info(f"Successfully removed existing container {container_name}")
        
        # Prepare environment variables and command
        environment = {
            "CUDA_VISIBLE_DEVICES": str(config["device_id"]),
        }
        if hf_mirror:
            environment["HF_MIRROR"] = hf_mirror
        
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
        if config.get("cont_batching", False):
            command.extend(["--cont-batching"])
        if config.get("flash_attn", False):
            command.extend(["--flash-attn"])

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
            command.extend(["--ctx-size", str(config["ctx_size"])])
            command.extend(["--parallel", str(config["parallel"])])
            if config.get("rope_scaling"):
                command.extend(["--rope-scaling", config["rope_scaling"]])
            if config.get("rope_freq_scale"):
                command.extend(["--rope-freq-scale", config["rope_freq_scale"]])
            
        volumes = {}
        
        # Mount Ollama repository
        if ollama_path and os.path.isdir(ollama_path):
            logger.info(f"Mounting Ollama repository: {ollama_path}")
            volumes[ollama_path] = {'bind': '/ollama_models', 'mode': 'ro'}
            
            # Adjust the model path in the command
            ollama_rel_path = os.path.relpath(model_file_path, ollama_path)
            command[1] = f"/ollama_models/{ollama_rel_path}"
            logger.info(f"Adjusted model path: {command[1]}")
        
        # Log the full command for debugging
        logger.info(f"Starting container {container_name} for model {ollama_model_name} on device {config['device_id']} with command: {' '.join(command)}")
        
        # Create and start the container
        container = docker_client.containers.run(
            docker_image,
            command=command,
            name=container_name,
            detach=True,
            environment=environment,
            ports=ports,
            volumes=volumes,
            runtime="nvidia"  # Remove if not using NVIDIA GPU
        )
        
        logger.info(f"Container {container_name} for model {ollama_model_name} started with ID: {container.id}")
        return True
        
    except docker.errors.APIError as e:
        logger.error(f"Error starting container {container_name} for model {ollama_model_name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error starting container {container_name} for model {ollama_model_name}: {str(e)}")
        return False

def wait_for_container_ready(model_name, model_configs, docker_client, timeout=60, health_check_interval=2):
    """Wait for the container to be ready, with timeout."""
    container_name = model_configs[model_name]["container_name"]
    if model_name not in model_configs:
        return False
    
    # Give the container a moment to initialize before checking
    logger.info(f"Waiting for initial container startup for {model_name}")
    import time
    time.sleep(health_check_interval)
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check container status
        exists, status = get_container_status(container_name, docker_client)
        if exists and status == 'running':
            logger.info(f"Container for {model_name} is ready")
            return True
        elif exists and status != 'running':
            logger.warning(f"Container for {model_name} exists but is in state: {status}")
            
        # Log progress during waiting
        elapsed = time.time() - start_time
        if elapsed > health_check_interval and elapsed % 10 < health_check_interval:  # Log every ~10 seconds
            # Check container logs to see if there are any issues
            try:
                container = get_container(container_name, docker_client, all=True)
                if container:
                    logs = container.logs(tail=10).decode('utf-8', errors='replace')
                    logger.info(f"Waiting for {model_name} to be ready... ({elapsed:.1f}s elapsed), container status: {status}")
                    logger.info(f"Recent container logs: {logs}")
            except Exception as e:
                logger.error(f"Error getting container logs: {str(e)}")
            
        time.sleep(health_check_interval)
    
    # If we timed out, get the container logs to help diagnose the issue
    try:
        container = get_container(container_name, docker_client, all=True)
        if container:
            logs = container.logs(tail=50).decode('utf-8', errors='replace')
            logger.error(f"Container logs for {model_name} after timeout: {logs}")
    except Exception as e:
        logger.error(f"Error getting container logs after timeout: {str(e)}")
    
    logger.error(f"Timeout waiting for {model_name} container to be ready")
    return False

def stop_all_containers(model_configs, docker_client):
    """Stop all containers in model_configs."""
    logger.info("Stopping all containers...")
    
    for model_name, config in model_configs.items():
        container_name = config["container_name"]
        
        # Check if container exists in any state
        exists, status = get_container_status(container_name, docker_client)
        if exists:
            logger.info(f"Found container {container_name} for model {model_name} in state: {status}")
            if stop_container(container_name, docker_client, remove=True):
                logger.info(f"Successfully stopped container {container_name}")
            else:
                logger.error(f"Failed to stop container {container_name}, attempting force removal")
                # If stopping fails, try direct removal
                if remove_container_if_exists(container_name, docker_client):
                    logger.info(f"Successfully removed container {container_name}")
                else:
                    logger.error(f"Failed to remove container {container_name}")
        else:
            logger.info(f"No container found for {model_name}")
    
    logger.info("All containers stopped") 