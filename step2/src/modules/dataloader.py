import torch
from torch.utils.data import DataLoader

def collate_fn(data):
    image_name, image_tensor, labels = zip(*data)

    # images
    image_tensor = torch.stack(image_tensor, 0)

    # stack labels
    labels = torch.stack(labels, dim=0)

    return image_name, image_tensor, labels

def prepare_dataloader(dataset, args, split):

    if split == "train":
        batch_size = args.batch_size_train
        shuffle    = True
    else:
        batch_size = args.batch_size_test
        shuffle    = False

    num_workers        = args.num_workers
    persistent_workers = args.persistent_workers

    dataloader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              num_workers=num_workers, 
                              persistent_workers=persistent_workers, collate_fn=collate_fn, shuffle=shuffle)
    
    return dataloader
