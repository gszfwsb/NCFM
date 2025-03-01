import argparse
import yaml
class ArgsProcessor:
    def __init__(self, config_path):
        self.config_path = config_path

    # Function to flatten a dictionary (without adding prefixes)
    def flatten_dict(self, d, parent_key='', sep='_'):
        """
        Recursively flattens a nested dictionary, but does not add the parent key.
        """
        items = []
        for k, v in d.items():
            new_key = k  # Use the current key directly, without adding the parent key
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Add the contents of the YAML configuration file to args
    def add_args_from_yaml(self, args):
        # Read the YAML configuration file
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Flatten the configuration dictionary
        flat_config = self.flatten_dict(config)

        # Convert value types (handle floating point numbers and booleans)
        for key, value in flat_config.items():
            # Convert to float if possible
            if isinstance(value, str):
                if value.lower() in ['true', 'false']:
                    flat_config[key] = value.lower() == 'true'
                elif 'e' in value or '.' in value:
                    try:
                        flat_config[key] = float(value)
                    except ValueError:
                        pass

        # Add the flattened configuration items to args
        for key, value in flat_config.items():
            setattr(args, key, value)

        return args

# Example usage
if __name__ == '__main__':
    # Create ArgumentParser instance
    parser = argparse.ArgumentParser(description='Configuration parser')

    # Add command line argument --config to pass the YAML file path
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file')

    # Parse command line arguments
    args = parser.parse_args()

    # Create Args_Processor instance and load the configuration
    args_processor = ArgsProcessor(args.config_path)
    args = args_processor.add_args_from_yaml(args)
    
    # Test output
    print(args.lr_sigma)  # Output: 0.005
    print(args.dataset)  # Output: cifar10/ipc1/sgd_lr0.01_20211212