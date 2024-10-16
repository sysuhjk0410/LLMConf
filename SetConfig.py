import yaml
import random
import csv
import os
from datetime import datetime

# Read config.yml
def load_config(filename='LLMConf/config.yml'):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)

# Randomly select a value
def get_random_value(param):
    param_type = param['type']
    if param_type == 'Enumeration':
        return random.choice(param['values'])
    elif param_type == 'Integer':
        return random.randint(param['range'][0], param['range'][1])
    elif param_type == 'Float':
        return round(random.uniform(param['range'][0], param['range'][1]), 2)
    elif param_type == 'Boolean':
        return random.choice(param['values'])
    else:
        raise ValueError(f"Unsupported type: {param_type}")

# Build command line instruction and generate configuration parameter values for each run
def build_command(config):
    command = [
        "vllm", "serve", "LLMConf/BaseLLM/Meta-Llama-3-8B-Instruct", "--port", "8100"
    ]
    
    config_values = {}

    # Iterate through each parameter in the config, generate random values and add them to the command
    for param_name, param_info in config.items():
        random_value = get_random_value(param_info)
        
        # If it is a Boolean type and the value is True, only add the parameter name; if False, ignore the parameter
        if param_info['type'] == 'Boolean':
            if random_value:
                command.append(f"--{param_name.replace('_', '-')}")
            config_values[param_name] = random_value
        else:
            command.append(f"--{param_name.replace('_', '-')}") 
            command.append(str(random_value))
            config_values[param_name] = random_value

    return command, config_values

# Save configuration parameters and their values to a CSV file (save 10 parameter values, excluding the index column)
def save_to_csv(config, config_values, filename='./data/data.csv'):
    file_exists = os.path.exists(filename)

    with open(filename, mode='a', newline='') as file:  # Use append mode 'a'
        writer = csv.writer(file)

        # If the file does not exist or is empty, write the header
        if not file_exists or os.stat(filename).st_size == 0:
            header = [param_name for param_name in list(config.keys())[:10]]  # Save 10 parameter columns
            writer.writerow(header)

        # Save the values of the first 10 configuration parameters
        row = [config_values.get(param_name, 'N/A') for param_name in list(config.keys())[:10]]
        writer.writerow(row)

def get_command():
    # Load config.yml
    config = load_config('LLMConf/config.yml')

    # Generate the command and the configuration parameter values
    command, config_values = build_command(config)
    
    # Save the configuration to a CSV file (10 parameter values)
    save_to_csv(config, config_values)

    return ' '.join(command)

# Execute the main function (only called for debugging)
if __name__ == "__main__":
    get_command()
