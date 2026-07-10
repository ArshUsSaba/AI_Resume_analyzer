import torch
import logging
import sys
import os
import ssl

def bypass_ssl_verification():
    """
    Globally bypasses SSL verification for standard urllib, requests, and httpx requests.
    This fixes certification check failures in proxy-restricted environments.
    """
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    try:
        import requests
        old_merge = requests.Session.merge_environment_settings
        def new_merge(self, url, headers, hooks, cookies, verify, cert, proxies):
            return old_merge(self, url, headers, hooks, cookies, False, cert, proxies)
        requests.Session.merge_environment_settings = new_merge
    except Exception:
        pass

    try:
        import httpx
        old_init = httpx.Client.__init__
        def new_init(self, *args, **kwargs):
            kwargs['verify'] = False
            old_init(self, *args, **kwargs)
        httpx.Client.__init__ = new_init
    except Exception:
        pass

# Run bypass immediately
bypass_ssl_verification()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AI-Resume-Analyzer")

def get_gpu_status():
    """
    Checks GPU availability via PyTorch and returns hardware metadata.
    Returns:
        dict: Containing keys 'available', 'device_count', 'device_name', 'device_type', 'acceleration_status'
    """
    status = {
        "available": False,
        "device_count": 0,
        "device_name": "CPU",
        "device_type": "cpu",
        "acceleration_status": "CPU (No CUDA acceleration available)",
        "vram_allocated_gb": 0.0,
        "vram_reserved_gb": 0.0
    }
    
    if torch.cuda.is_available():
        status["available"] = True
        status["device_count"] = torch.cuda.device_count()
        status["device_name"] = torch.cuda.get_device_name(0)
        status["device_type"] = "cuda"
        status["acceleration_status"] = "CUDA Enabled (NVIDIA GPU Accelerated)"
        
        # Get memory metrics if possible
        try:
            status["vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
            status["vram_reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2)
        except Exception:
            pass
            
    return status

def log_info(message):
    logger.info(message)

def log_error(message, exc=None):
    if exc:
        logger.error(f"{message} - Error: {str(exc)}", exc_info=True)
    else:
        logger.error(message)

