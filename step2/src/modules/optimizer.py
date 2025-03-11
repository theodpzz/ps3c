import torch
import numpy as np
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

def get_trainable_parameters(args, model):

    return model.parameters()

def get_scheduler(args, optimizer, num_batches):

    # constant schedule with wamrup
    if(args.name_scheduler == "constant"):
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps = args.num_warmup_steps)
    
    # cosine schedule with wamrup
    if(args.name_scheduler == "cosine"):
        args.num_training_steps = args.epochs * num_batches
        return get_cosine_schedule_with_warmup(optimizer, 
                                               num_warmup_steps   = args.num_warmup_steps,
                                               num_training_steps = args.num_training_steps)
    
def get_optimizer_scheduler(args, model, num_batches=None):

    # extract trainable parameters
    trainable_parameters = get_trainable_parameters(args, model)

    # optimizer
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    
    # scheduler
    lr_scheduler = get_scheduler(args, optimizer, num_batches)

    return optimizer, lr_scheduler
