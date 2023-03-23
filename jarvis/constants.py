import os
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="Jarvis command-line tool")

# Define the --config-path argument
parser.add_argument(
    "--config-path",
    type=str,
    required=True,  # Set to 'False' if the argument is optional
    help="Path to the configuration file",
)

# Parse the command-line arguments
args = parser.parse_args()

# Access the --config-path argument value
config_path = args.config_path

ENV_PATH = os.path.join(config_path, ".env")
MATERIAL_FILE = os.path.join(config_path, "data", "material.csv")
SOURCE_FILE = os.path.join(config_path, "data", "source.csv")
STORED_TOKEN = os.path.join(config_path, "auth", "token.json")
CREDENTIAL_TOKEN = os.path.join(config_path, "auth", "credentials.json")
TEMPLATE_FOLDER = os.path.join(config_path, "templates")
