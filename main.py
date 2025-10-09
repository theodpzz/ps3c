"""
##### Training
### For Step 1
python main.py train --yaml_file ./step1/configs/train/swinv2.yaml --step 1
python main.py train --yaml_file ./step1/configs/train/seresnext.yaml --step 1
python main.py train --yaml_file ./step1/configs/train/convnextv2.yaml --step 1

### For Step 2
python main.py train --yaml_file ./step2/configs/train/swinv2.yaml --step 2
python main.py train --yaml_file ./step2/configs/train/seresnext.yaml --step 2
python main.py train --yaml_file ./step2/configs/train/convnextv2.yaml --step 2

##### Inference
### For Step 1
python main.py infer --yaml_file ./step1/configs/inference/swinv2.yaml --step 1
python main.py infer --yaml_file ./step1/configs/inference/seresnext.yaml --step 1
python main.py infer --yaml_file ./step1/configs/inference/convnextv2.yaml --step 1

### For Step 2
python main.py infer --yaml_file ./step2/configs/inference/swinv2.yaml --step 2
python main.py infer --yaml_file ./step2/configs/inference/seresnext.yaml --step 2
python main.py infer --yaml_file ./step2/configs/inference/convnextv2.yaml --step 2
"""

import argparse

from step1.train import train_step_1
from step2.train import train_step_2
from step1.infer import infer_step_1
from step2.infer import infer_step_2

def train(yaml_file, step):

    if step == 1:
        train_step_1(yaml_file)

    elif step == 2:
        train_step_2(yaml_file)
        
def infer(yaml_file, step):
    if step == 1:
        infer_step_1(yaml_file)
    elif step == 2:
        infer_step_2(yaml_file)
    
if __name__ == "__main__":
    
    # read config file
    parser     = argparse.ArgumentParser(description='Main entry point for all commands')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Train parser
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--yaml_file', type=str, default='configs/train/default.yaml', help='YAML file containing parameters')
    train_parser.add_argument('--step', type=int, default=1, help='Train step')

    # Inference parser
    infer_parser = subparsers.add_parser('infer', help='Infer a model')
    infer_parser.add_argument('--yaml_file', type=str, default='configs/inference/default.yaml', help='YAML file containing parameters')
    infer_parser.add_argument('--step', type=int, default=1, help='Infer step')

    # Parser
    args = parser.parse_args()
    yaml_file = args.yaml_file
    step      = args.step

    # run script
    if args.command == "train":
        train(yaml_file, step)

    elif args.command == "infer":
        infer(yaml_file, step)