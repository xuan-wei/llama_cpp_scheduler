import subprocess
import socket
from pathlib import Path
import docker
from concurrent.futures import ThreadPoolExecutor

import logging

def setup_logger(name):
    """Configure and return a logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name) 

logger = setup_logger('scheduler_utils')

def get_best_gpu(force_gpu_id=None):
    """
    Returns the ID of the GPU with the most available memory using nvidia-smi.
    """
    if force_gpu_id is not None:
        return force_gpu_id
        
    try:
        # Run nvidia-smi to get memory info
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'])
        memory_free_list = [int(x) for x in output.decode('utf-8').strip().split('\n')]
        
        if not memory_free_list:
            return 0  # Default to GPU 0 if no info available
            
        return memory_free_list.index(max(memory_free_list))
        
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return 0  # Default to GPU 0 if command fails

def find_next_available_port(start_port=8090, max_port=9000):
    """Find the next available port starting from start_port."""
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Setting a timeout to speed up the check
            sock.settimeout(0.1)
            # If we can't connect to the port, it's available
            result = sock.connect_ex(('127.0.0.1', port))
            if result != 0:  # Port is available
                return port
    # If no ports are available in the range
    raise RuntimeError(f"No available ports in range {start_port}-{max_port}")

def download_model(model_name, download_url, model_base_path):
    """Download model using axel, wget, or curl if it doesn't exist."""
    model_path = Path(model_base_path) / f"{model_name}.gguf"
    if model_path.exists():
        logger.info(f"Model {model_name} already exists at {model_path}")
        return True

    # Check if download URL is empty or None
    if not download_url:
        logger.warning(f"No download URL provided for model {model_name}, skipping download")
        return False

    temp_path = Path(model_base_path) / f"{model_name}.gguf.downloading"
    logger.info(f"Downloading model {model_name} from {download_url}")
    
    # Try different download methods in order of preference
    download_methods = [
        {
            "name": "axel",
            "check_cmd": ["axel", "--version"],
            "download_cmd": ["axel", "-n", "16", "-o", str(temp_path), download_url]
        },
        {
            "name": "wget",
            "check_cmd": ["wget", "--version"],
            "download_cmd": ["wget", "--progress=bar:force", "-O", str(temp_path), download_url]
        },
        {
            "name": "curl",
            "check_cmd": ["curl", "--version"],
            "download_cmd": ["curl", "-L", "-o", str(temp_path), download_url, "--progress-bar"]
        }
    ]
    
    # Find the first available download method
    download_method = None
    for method in download_methods:
        try:
            subprocess.run(method["check_cmd"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            download_method = method
            logger.info(f"Using {method['name']} for downloading")
            break
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    
    if not download_method:
        logger.error("No download method available. Please install axel, wget, or curl.")
        return False
    
    try:
        # Run the download command and capture output
        process = subprocess.Popen(
            download_method["download_cmd"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # Show download progress
        last_percentage = -1
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
                
            if output:
                # Different progress parsing for different tools
                if download_method["name"] == "axel" and '[' in output and '%]' in output:
                    try:
                        current = int(output[output.find('[')+1:output.find('%')])
                        if current != last_percentage:  # Avoid duplicate percentages
                            speed = output.split('[')[-1].split('KB/s')[0].strip()
                            print(f"\rDownloading {model_name}: {current}% [{speed}KB/s]", end='', flush=True)
                            last_percentage = current
                    except (ValueError, IndexError):
                        pass
                elif download_method["name"] == "wget" and "%" in output:
                    try:
                        # Parse wget output like "45% [=======>      ] 123.45K/s eta 1m"
                        current = int(output.split('%')[0].strip())
                        if current != last_percentage:
                            print(f"\rDownloading {model_name}: {current}%", end='', flush=True)
                            last_percentage = current
                    except (ValueError, IndexError):
                        pass
                elif download_method["name"] == "curl":
                    # curl with --progress-bar shows a progress bar, just print the output
                    print(f"\r{output}", end='', flush=True)

        print()  # Print newline after download completes
        process.wait()  # Wait for the process to complete

        # Check if download was successful
        if process.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 0:
            # Rename the downloaded file
            temp_path.rename(model_path)
            logger.info(f"Successfully downloaded {model_name}")
            return True
        else:
            logger.error(f"Failed to download {model_name}: Process returned {process.returncode}")
            if temp_path.exists():
                temp_path.unlink()  # Clean up failed download
            return False

    except Exception as e:
        logger.error(f"Unexpected error downloading model {model_name}: {str(e)}")
        if temp_path.exists():
            temp_path.unlink()
        return False

def ensure_models_exist(models_config, model_base_path):
    """Ensure specified models exist, downloading if necessary."""
    failed_models = []
    with ThreadPoolExecutor(max_workers=1) as executor:  # Changed to 1 since we're downloading one at a time
        futures = {}
        for model_name, model_config in models_config.items():
            download_url = model_config.get('download_url', '')
            future = executor.submit(download_model, model_name, download_url, model_base_path)
            futures[future] = model_name
        
        # Wait for all downloads to complete and collect failures
        for future in futures:
            model_name = futures[future]
            try:
                if not future.result():
                    failed_models.append(model_name)
            except Exception as e:
                logger.error(f"Error downloading {model_name}: {str(e)}")
                failed_models.append(model_name)
    
    # Report failed downloads
    if failed_models:
        logger.warning(f"Failed to download the following models: {', '.join(failed_models)}")
        return False
    return True

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