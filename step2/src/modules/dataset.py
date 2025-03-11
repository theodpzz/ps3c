import os
import glob
import torch
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial

def getDatasets(args):

    # path with volumes
    path_images_train = args.path_images_train
    path_images_valid = args.path_images_train
    path_images_test  = args.path_images_train

    # path with labels
    path_labels = args.path_labels

    # path with anomaly binary labels
    path_labels           = os.path.join(args.path_labels, f"folder_{args.fold}")
    path_labels_train     = os.path.join(path_labels, "train.csv")
    path_labels_valid     = os.path.join(path_labels, "valid.csv")
    path_labels_test      = os.path.join(path_labels, "test.csv")

    # datasets
    train = APACCDataset(args, path_images_train, path_labels_train, 
                         batch_size=args.batch_size_train, split="train")
    valid = APACCDataset(args, path_images_valid, path_labels_valid, 
                         batch_size=args.batch_size_valid, split="valid")
    test  = APACCDataset(args, path_images_test, path_labels_test, 
                         batch_size=args.batch_size_test, split='test')

    return train, valid, test


class APACCDataset(Dataset):
    def __init__(self, args, path_images, path_labels, 
                 batch_size, split, resize_dim=256):

        # device
        self.device = torch.device('cpu')

        # important features
        self.args          = args
        self.split         = split
        self.path_images   = path_images
        self.path_labels   = path_labels
        self.dataset       = args.dataset
        self.batch_size    = batch_size
        self.num_classes   = args.num_classes
        self.img_size      = args.img_size

        # paths of nii files
        self.samples = self.prepare_samples_apacc()

        # read labels
        self.labels        = pd.read_csv(self.path_labels)

        # remove rubbish images for this task
        self.labels        = self.labels.loc[self.labels['label'].isin(['healthy', 'unhealthy', 'bothcells'])]

        # extract name of labels
        self.samples_names = list(self.labels['image_name'])

        # considered classes : healthy and-or unhealthy
        self.labels_names  = ['healthy', 'unhealthy']

        # filter to conserve samples from the split
        self.samples = [x for x in self.samples if os.path.basename(x) in self.samples_names]
        
        # dataset statistics
        if(self.args.normalization == "apacc"):
            self.mean = (0.54366329, 0.69260935, 0.68979313)
            self.std  = (0.35946042, 0.27779834, 0.26446231)
        elif(self.args.normalization == "imagenet"):
            self.mean = (0.485, 0.456, 0.406)
            self.std  = (0.229, 0.224, 0.225)

        # transform
        if(split == "train"):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([transforms.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.0),)], p=args.proba_crop),
                transforms.RandomVerticalFlip(p=args.proba_flip),
                transforms.RandomHorizontalFlip(p=args.proba_flip),
                transforms.RandomApply([transforms.ElasticTransform(alpha=(50.0, 150.0), sigma=5.0, fill=1.00)], p=args.proba_elastic),
                transforms.RandomApply([transforms.RandomRotation(20, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)], p=args.proba_rotate),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])            
        self.png_to_tensor = partial(self.png_img_to_tensor, transform = self.transform)

        # label to index
        self.label_to_index = {
                                "healthy":   0,
                                "unhealthy": 1,
                                "bothcells": [0, 1],
                                }

        self.log_dataset()

    def log_dataset(self):
        print(f'\n>>> Details about the {self.split} set:')
        print(f'    > Number of samples: {len(self.samples)} | batches: {round(len(self.samples)/self.batch_size)} | Batch size: {self.batch_size}\n')

    def prepare_samples_apacc(self):
        samples = []

        # iterate over images
        for nii_file in glob.glob(os.path.join(self.path_images, '*.png')):
            samples.append(nii_file)

        return samples

    def __len__(self):
        return len(self.samples)

    def png_img_to_tensor(self, path_image, transform):

        # load volume
        img = Image.open(path_image)

        # Convert to float32 for better precision and conserve 3 first channels
        img = np.array(img).astype(np.float32)[:, :, :3] / 255.0

        # apply transforms
        tensor = transform(img)

        return tensor

    def get_labels(self, path_img_file):
        
        # extract id of patient
        image_name = os.path.basename(path_img_file)

        # extract label as string
        label_str = self.labels[self.labels.image_name == image_name].label.values[0]

        # create a one-hot tensor
        index         = self.label_to_index[label_str]
        labels        = torch.zeros(self.num_classes)
        labels[index] = 1

        return image_name, labels

        
    def __getitem__(self, index):

        # get path of nii_file
        path_img_file = self.samples[index]

        # load volume and perform transformation
        image_tensor = self.png_to_tensor(path_img_file)

        # load labels
        image_name, labels = self.get_labels(path_img_file)

        return image_name, image_tensor, labels
      
