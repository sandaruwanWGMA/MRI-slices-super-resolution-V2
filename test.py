import json
import os

# Configuration data
config_data = {
    "input_dimensions": [3, 16, 112, 112],
    "output_dimensions": 400,
    "batch_size": 10,
    "epochs": 50,
    "learning_rate": 0.001,
    "dataset_path": "data/datasets",
    "model_save_path": "data/models",
    "log_path": "data/logs",
}

# Ensure the Options directory exists
options_dir = "Options"
os.makedirs(options_dir, exist_ok=True)

# Path to the configuration file
config_path = os.path.join(options_dir, "config.opt")

# Writing JSON data
with open(config_path, "w") as config_file:
    json.dump(config_data, config_file, indent=4)

print(f"Configuration file created at {config_path}")
