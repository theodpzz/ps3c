import os
import yaml

from datetime import datetime

def parse_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def make_output_dir(args):

    output_dir = args.save_dir
    output_dir = os.path.join(output_dir, datetime.now().strftime("%Y-%m-%d_%Hh%M"))
    os.makedirs(output_dir, exist_ok=True)

    path_checkpoints   = os.path.join(output_dir, "checkpoints")
    path_logs          = os.path.join(output_dir, "logs")
    path_figures       = os.path.join(output_dir, "figures")

    os.makedirs(path_checkpoints, exist_ok=True)
    os.makedirs(path_logs, exist_ok=True)
    os.makedirs(path_figures, exist_ok=True)

    args.save_dir = output_dir

    return args

