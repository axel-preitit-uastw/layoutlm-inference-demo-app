import json


# This module provides a function to read a configuration file in JSON format.
def get_json_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config
