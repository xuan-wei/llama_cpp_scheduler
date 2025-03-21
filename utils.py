import subprocess
import socket
from pathlib import Path
import docker
from concurrent.futures import ThreadPoolExecutor
import json
import os
import logging
import requests

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

def parse_ollama_model_id(model_name):
    """Parse Ollama model name into namespace, model and tag parts."""
    parts = model_name.split(':')
    model_id = parts[0]
    tag = parts[1] if len(parts) > 1 else 'latest'
    
    if '/' in model_id:
        namespace, model = model_id.split('/', 1)
    else:
        namespace = 'library'
        model = model_id
        
    return namespace, model, tag

def find_ollama_model_manifest(ollama_path, model_name, is_loading_dashboard = True):
    """Find the manifest file for an Ollama model."""
    try:
        namespace, model, tag = parse_ollama_model_id(model_name)
        
        # Handle special case for models like 'qwen2.5:14b'
        if model == 'qwen2.5' and tag == '14b':
            manifest_path = os.path.join(ollama_path, 'manifests', 'registry.ollama.ai', namespace, model, tag)
        else:
            # First try with tag as subdirectory
            manifest_path = os.path.join(ollama_path, 'manifests', 'registry.ollama.ai', namespace, model, tag)
            if not os.path.exists(manifest_path):
                # Try with tag added to model name
                manifest_path = os.path.join(ollama_path, 'manifests', 'registry.ollama.ai', namespace, f"{model}-{tag}")
        
        if os.path.exists(manifest_path):
            if is_loading_dashboard:
                logger.debug(f"Found manifest at {manifest_path}")
            else:
                logger.info(f"Found manifest at {manifest_path}")
            with open(manifest_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Manifest not found at {manifest_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error finding manifest for model {model_name}: {str(e)}")
        return None

def find_ollama_model_file(ollama_path, model_name, is_loading_dashboard = True):
    """Find the GGUF file for a model in the Ollama repository."""
    manifest = find_ollama_model_manifest(ollama_path, model_name, is_loading_dashboard)
    if not manifest:
        logger.error(f"Could not find manifest for model {model_name}")
        return None
        
    try:
        # Look for model layer
        model_digest = None
        for layer in manifest.get('layers', []):
            if layer.get('mediaType') == 'application/vnd.ollama.image.model':
                model_digest = layer.get('digest')
                break
                
        if not model_digest:
            logger.error(f"No model layer found in manifest for {model_name}")
            return None
            
        # Extract the SHA256 hash
        if model_digest.startswith('sha256:'):
            sha256_hash = model_digest.split(':', 1)[1]
            model_blob_path = os.path.join(ollama_path, 'blobs', f'sha256-{sha256_hash}')
            
            if os.path.exists(model_blob_path):
                if is_loading_dashboard:
                    logger.debug(f"Found model file at {model_blob_path}")
                else:
                    logger.info(f"Found model file at {model_blob_path}")
                return model_blob_path
            else:
                logger.error(f"Model blob not found at {model_blob_path}")
                return None
        else:
            logger.error(f"Invalid digest format: {model_digest}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting model path from manifest for {model_name}: {str(e)}")
        return None

def is_ollama_model_available(ollama_path, model_name, is_loading_dashboard = True):
    """Check if a model is available in the Ollama repository."""
    if not ollama_path or not os.path.isdir(ollama_path):
        logger.warning(f"Ollama path {ollama_path} is not a valid directory")
        return False
        
    model_file = find_ollama_model_file(ollama_path, model_name, is_loading_dashboard)
    return model_file is not None

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

def is_container_running(port, container_name):
    """Check if a container is running and ready to accept requests using Docker SDK."""
    try:

        # Try to connect to the health check endpoint
        try:
            response = requests.get(f'http://localhost:{port}/health', timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.debug(f"Container {container_name} is not ready yet: {str(e)}")
            return False
            
    except docker.errors.APIError as e:
        logger.error(f"Error checking if container {container_name} is running: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking container {container_name}: {str(e)}")
        return False

def get_container(container_name, docker_client, all=True):
    """Get a container by name, including stopped containers if all=True."""
    try:
        containers = docker_client.containers.list(filters={"name": container_name}, all=all)
        return containers[0] if containers else None
    except docker.errors.APIError as e:
        logger.error(f"Error getting container {container_name}: {str(e)}")
        return None 