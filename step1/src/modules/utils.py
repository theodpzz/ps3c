import os
import json
import yaml
import argparse

def parse_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def make_output_dir(args):

    output_dir       = os.path.join(args.save_dir, f"folder_{args.fold}")
    os.makedirs(output_dir, exist_ok=True)

    path_checkpoints   = os.path.join(output_dir, "checkpoints")
    path_logs          = os.path.join(output_dir, "logs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(path_checkpoints, exist_ok=True)
    os.makedirs(path_logs, exist_ok=True)

    return output_dir

