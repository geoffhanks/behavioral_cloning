import pandas as pd
import numpy as np

import os

import torch
import torchvision
from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt


class BCDataset(Dataset):
    """
    Custom pytorch dataloader for MLP
    """

    def __init__(self, data_file, img_dir, transform=None):
        """
        Intialize the data loader
        """

        # get the file with the list of images and annotations
        self.data = pd.read_csv(data_file, header=None)

        # set image directory
        self.img_dir = img_dir

        # set the transform
        self.transform = transform
        

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets an image at the index idx

        Args 
            idx Index of the data sample
        Returns
            image, label The image at the specified index with the apply transform and its corresponding label from the 
            annotations file
        """

        # get the image path from the image directory and image name
        img_path = os.path.join(self.img_dir, "image" + str(int(self.data.iloc[idx, 0])) + ".jpeg")

        # Read the image and get the label
        image = (read_image(img_path))/255 # divide by 255 to transform uint8 to a float between 0 and 1
        joint_data = torch.tensor(self.data.iloc[idx, 2:9].values)
        label = torch.tensor(self.data.iloc[idx, 9:].values)

        # apply the tranform if applicable
        if self.transform:
            image = self.transform(image)

        # return the image and label
        return image, joint_data, label

def main():
    # test the data loader by displaying images and their labels in the windo title
    train_dataset = BCDataset("./Data/testfile.csv", 
                                  "/home/airlab4/ros_ws/src/shared_control/data/trajectory_BC", None)
    
    train_dataLoader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

    while(True):

        train_images, joint_data, train_labels = next(iter(train_dataLoader))

        print(f"Feature batch shape: {train_images.size()}")
        print(f"Joint data size {joint_data.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_images[4].squeeze().permute(1, 2, 0)
        label = train_labels[4]
        plt.imshow(img, cmap="gray")
        print(f"Label: {label}")
        plt.show()
        # input()



if  __name__ == "__main__":
    main()