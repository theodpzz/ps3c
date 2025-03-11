import torch
from torch.utils.data import DataLoader

def collate_fn(data):
    image_name, image_tensor, labels = zip(*data)

    # images
    image_tensor = torch.stack(image_tensor, 0)

    # stack labels
    labels = torch.stack(labels, dim=0)

    return image_name, image_tensor, labels

def prepare_dataloader(trainset, validset, testset, args):

    train_loader = DataLoader(trainset, batch_size=args.batch_size_train, 
                                num_workers=args.num_workers, 
                                persistent_workers=args.persistent_workers, collate_fn=collate_fn, shuffle=True)
    
    valid_loader = DataLoader(validset, batch_size=args.batch_size_valid, 
                                num_workers=args.num_workers_valid, 
                                persistent_workers=args.persistent_workers_val, collate_fn=collate_fn, shuffle=False)
    
    test_loader  = DataLoader(testset, batch_size=args.batch_size_test, 
                                num_workers=args.num_workers_test, 
                                persistent_workers=args.persistent_workers_test, collate_fn=collate_fn, shuffle=False)

    return train_loader, valid_loader, test_loader
