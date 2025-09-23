import time
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login

def timed(func):
    """Decorator that prints the runtime of any function it wraps."""
    def wrapper(*args, **kwargs):  
        print(f"{func.__name__} starts runing!")          
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - t0:.4f}s")
        return out
    return wrapper  

def best_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
        
    return torch.float32

def best_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def login_huggingface():
    load_dotenv()
    login(os.getenv("HUGGINGFACE_KEY"))
    print("Login HuggingFace!")  

__all__ = ["timed", "best_dtype", "best_device"]