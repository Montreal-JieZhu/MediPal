import time
import torch
import os
from typing import Any
from dotenv import load_dotenv
from huggingface_hub import login
from IPython.display import Image, display
from langgraph.graph.state import CompiledStateGraph

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
    """Return the best dtype for the device"""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
        
    return torch.float32

def best_device():
    """Return the device type"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def login_huggingface():
    """Login HaggingFace"""
    load_dotenv()
    login(os.getenv("HUGGINGFACE_KEY"))
    print("Login HuggingFace!")  

def draw_langgraph(app: CompiledStateGraph, inline: bool = True):
    """Draw the graph for langgraph application
       Defaul Can be inline graph for jupyter notebook.
       Can output a png to current folder.
    """
    if inline:
        display(Image(data=app.get_graph().draw_mermaid_png()))
    else:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open("langgraph_diagram.png", "wb") as f:
            f.write(png_bytes)


__all__ = ["timed", "best_dtype", "best_device", "draw_langgraph"]