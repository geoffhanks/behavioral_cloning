#!/usr/bin/env python3

import torch
from torch import nn

from typing import Tuple, Union, List, Callable
from torch.optim import SGD
import torchvision
from torchvision.transforms import Resize
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from Model import BCModel
from NullspaceModel import NullspaceModel, NullspaceModel_Norm, NullspaceOnlyModel_Norm, NullspaceOnlyModel_Dropout

import json

# from BCDataset import BCDataset
from NullspaceDataset import NullSpaceOnlyDataset, BCDataset
import os


# import tikzplotlib



class BehaviouralCloning():

    def __init__(self) -> None:
        """
        Initialize class.  sets up gpu if available, loads data set with batch size
        """
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.DEVICE)  # this should print out CUDA

        self.load_data(32)   

        # print(self.val_loader.shape)

    def load_data(self, batch_size=128):
        """
        Loads the data sets and wraps them in batch loaders with given size.  Creates test, train val, debug_train and debug_val sets

        Args:
            batch size : int representing the sizes of the desired batches.  defaults to 128
        
        """


        self.resize_width = 80
        self.resize_height = 150
        self.channels = 3

        self.batch_size = batch_size


        # create the custom data sets
        self.train_dataset = NullSpaceOnlyDataset("./Data/nullspace_and_stiffness_BC/trainfile9.csv", 
                                  "/home/airlab4/ros_ws/src/shared_control/data/nullspace_and_stiffness_BC/dataset9", None)
        
        self.val_dataset = NullSpaceOnlyDataset("./Data/nullspace_and_stiffness_BC/valfile9.csv", 
                                  "/home/airlab4/ros_ws/src/shared_control/data/nullspace_and_stiffness_BC/dataset9", None)
        

        self.test_dataset = NullSpaceOnlyDataset("./Data/nullspace_and_stiffness_BC/testfile9.csv", 
                                  "/home/airlab4/ros_ws/src/shared_control/data/nullspace_and_stiffness_BC/dataset9", None)
        
        # create data loaders

        self.train_loader = DataLoader(self.train_dataset, self.batch_size, True)
        self.val_loader = DataLoader(self.val_dataset, self.batch_size, True)
        self.test_loader = DataLoader(self.test_dataset, self.batch_size, True)

 


        # # # data sets using 10% of train data for debugging
        # self.debug_train_dataset, self.debug_val_dataset = random_split(self.val_dataset, [round(0.9 * len(self.val_dataset)), round(0.1 * len(self.val_dataset))])

        # # create debugging dataloaders
        # self.debug_train_loader = DataLoader(
        #     self.debug_train_dataset,
        #     batch_size=batch_size,
        #     shuffle=True
        # )

        # self.debug_val_loader = DataLoader(
        #     self.debug_val_dataset,
        #     batch_size=batch_size,
        #     shuffle=True
        # )

        
    def train(  self, 
                model: nn.Module, optimizer: SGD,
                train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 20
                )-> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Trains a model for the specified number of epochs using the loaders.

        Args:
            model: a model to train
            optimizer: SGD optimizer initialized with model parameters and hyperpraameters
            train_loader : Pytorch dataloader with training data
            val_data : pytorch dataloader with validation data
            epochs: int with number of epochs to train.  defaults to 20

        Returns: 
            Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.
        """

        # create loss function
        # loss = nn.BCEWithLogitsLoss()
        loss = nn.MSELoss()
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        train_loss = 0.0
        train_acc = 0.0


        # progress bar based on epochs
        with tqdm(total=epochs, leave=False) as pbar_train: 

            # iterate over epochs
            for e in range(epochs):
                # set progress bar desc
                pbar_train.set_description(f"Epoch {e},  acc: {train_loss/len(train_loader):.5f}")  #/ (self.batch_size * len(train_loader))

                # set the mode to train mode
                model.train()

                # initialize loss and accuracy
                train_loss = 0.0
                train_acc = 0.0

                # Main training loop; iterate over train_loader. The loop
                # terminates when the train loader finishes iterating, which is one epoch.
                for (image, train_joint_states, labels) in train_loader:
                    # labels = labels.squeeze()

                    # get current batch
                    image, train_joint_states, labels = image.to(self.DEVICE), train_joint_states.to(self.DEVICE), labels.to(self.DEVICE)
                    # zero optimizer graidents
                    optimizer.zero_grad()
                    #get predictions from model
                    labels_pred = model(image, train_joint_states).squeeze()
                    # calc loss
                    batch_loss = loss(labels_pred, labels.float())
                    # sum loss
                    train_loss = train_loss + batch_loss.item()

                    # get predictions base on highest logit
                    # labels_pred_max = (labels_pred > 0).int() #torch.argmax(labels_pred, 1)
                    # get accuracy
                    # batch_acc = torch.sum(labels_pred_max == labels)
                    #sum accuracy
                    # train_acc = train_acc + batch_acc.item()

                    #perform backward pass
                    batch_loss.backward()
                    # step with optimizer
                    optimizer.step()

                # append losses and accuracies for this training epoch
                train_losses.append(train_loss / len(train_loader))
                # train_accuracies.append(train_acc / (self.batch_size * len(train_loader)))

                # Validation loop; use .no_grad() context manager to save memory.
                model.eval()
                val_loss = 0.0
                val_acc = 0.0

                with torch.no_grad():
                    # iterate over validation set
                    for (v_batch, val_joint_states, labels) in val_loader:
                        # labels = labels.squeeze()
                        # get current val batch
                        v_batch, val_joint_states, labels = v_batch.to(self.DEVICE), val_joint_states.to(self.DEVICE), labels.to(self.DEVICE)
                        # forward pass
                        labels_pred = model(v_batch, val_joint_states)#.squeeze()
                        # get loss, add to running total
                        v_batch_loss = loss(labels_pred, labels.float())
                        val_loss = val_loss + v_batch_loss.item()

                        # make class prediction based on highest logit
                        # v_pred_max = (labels_pred > 0).int()# torch.argmax(labels_pred, 1)
                        # get accuracy and add to running total
                        # batch_acc = torch.sum(v_pred_max == labels)
                        # val_acc = val_acc + batch_acc.item()

                    # add epoch val loss and accuracy
                    val_losses.append(val_loss / len(val_loader))
                    # val_accuracies.append(val_acc / (self.batch_size * len(val_loader)))

                 # update progress bar   
                pbar_train.update()
        # return losses and accuracies
        return train_losses, val_losses#train_accuracies, val_losses, val_accuracies
    
    def evaluate(self,
            model: nn.Module, loader: DataLoader
        ) -> Tuple[float, float]:
        """Computes test loss and accuracy of model on loader.
        
        Args: 
            model: model to be evaluated
            loader: pytorch dataloader with testing data

        returns:
            Tuple: test loss and accuracy
        
        """

        # create loss type
        # loss = nn.BCEWithLogitsLoss()
        loss = nn.MSELoss()


        # set model to evaluate mode, init accuracy and loss
        model.eval()
        test_loss = 0.0
        test_acc = 0.0

        # torch.no_grad to save memory
        with torch.no_grad():
            #iterate over batches
            for (batch, joint_states, labels) in loader:
                # get current batch data, send to gpu
                batch, joint_states, labels = batch.to(self.DEVICE), joint_states.to(self.DEVICE), labels.to(self.DEVICE)

                #forward pass
                y_batch_pred = model(batch, joint_states)#.squeeze()
                #get loss and add to runnning total
                batch_loss = loss(y_batch_pred, labels.float())
                test_loss = test_loss + batch_loss.item()

                # get accuracy and add to running total
                # pred_max = torch.argmax(y_batch_pred, 1)
                # pred_max = (y_batch_pred > 0).int()
                # batch_acc = torch.sum(pred_max == labels)
                # test_acc = test_acc + batch_acc.item()

            # average loss and accuracy
            test_loss = test_loss / len(loader)
            # test_acc = test_acc / (self.batch_size * len(loader))
        # return desired params
        return test_loss#, test_acc
            


def main():

    # create trainign object
    mlp = BehaviouralCloning()

    # create model object of correct type
    model = NullspaceOnlyModel_Dropout().to(mlp.DEVICE)
    # model = NullspaceOnlyModel_Norm().to(mlp.DEVICE)
    # model.load_state_dict(torch.load("./Models/nullspace_and_stiffness_BC/Model9-1.pt"))
    # optim = SGD(model.parameters(), 0.08, 0.3) # lr for dropout not batch norm

    # create the model optimizer
    optim = SGD(model.parameters(), 0.08, 0.3)

    # runthe training function
    train_loss, val_loss = mlp.train(model, optim, mlp.train_loader, mlp.val_loader, epochs=150)


    # evaluate the model
    test_loss = mlp.evaluate(model, mlp.test_loader)


    # display the evaluation loss
    print(test_loss)

    # save model weights
    torch.save(model.state_dict(), "./Models/nullspace_and_stiffness_BC/Model9-8.pt")

    # create dict with training and validation losses
    data = {
        "train_loss": train_loss,
        "val_loss" : val_loss
    }

    # save the training and validation losses
    with open("./training_curves9-8.json", 'w') as f:
        f.write(json.dumps(data))


    # show plot of training and validation losses
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.grid(True)
    plt.legend()
    plt.title("Trainig and Validation Losses Over Training Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.show()

if __name__ == "__main__":
    main()