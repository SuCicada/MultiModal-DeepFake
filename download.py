from datasets import load_dataset
import os

# Set custom cache directory - you can modify this path as needed
cache_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(cache_dir, exist_ok=True)

dataset = load_dataset("rshaojimmy/DGM4", split="test", cache_dir=cache_dir)