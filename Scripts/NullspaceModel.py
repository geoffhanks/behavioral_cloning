#!/usr/bin/env python3
import torch
from torch import nn



class NullspaceModel(torch.nn.Module):
    """
    Model used to predict nullspace configuration and path planning with dropout regularization
    """
    def __init__(self):
        super().__init__()


        # set parameters
        self.dropout = 0.00
        self.dropout2d = 0.00


        ## image processing model
        self.num_channels1 = 10
        self.num_channels2 = 20
        self.kernel1_size = 5
        self.kernel2_size = 5

        self.max_pool_size = 8
        self.image_lin_out_size = 150

        self.conv1_out_size = (200 - (self.kernel1_size-1))
        self.conv2_out_size = (self.conv1_out_size - (self.kernel2_size-1) )
        self.max_pool_out_size = torch.square(
            torch.floor(torch.tensor((self.conv2_out_size - (self.max_pool_size - 1)-1)/self.max_pool_size + 1)).int())
        
        self.image_processing = nn.Sequential(
            nn.Conv2d(3, self.num_channels1, self.kernel1_size),
            nn.ReLU(),
            nn.Dropout2d(self.dropout2d),
            nn.Conv2d(self.num_channels1, self.num_channels2, self.kernel2_size),
            nn.ReLU(),
            nn.Dropout2d(self.dropout2d),
            nn.MaxPool2d(self.max_pool_size),
            nn.Flatten(),
            nn.Linear(self.max_pool_out_size.item()*self.num_channels2, self.image_lin_out_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.image_lin_out_size, self.image_lin_out_size),
        )


        ## joint state processing layer
        self.joint_lin1_out = 75
        self.joint_lin2_out = 75


        self.joint_state_processing = nn.Sequential(
            nn.Linear(7, self.joint_lin1_out).to(torch.float),
            nn.ReLU().to(torch.float),
            nn.Dropout(self.dropout),
            nn.Linear(self.joint_lin1_out, self.joint_lin2_out).to(torch.float),
        )

        ## combined processing layers

        self.combined_hidden_size = 100

        self.combine = nn.Sequential(
            nn.Linear(self.image_lin_out_size + self.joint_lin2_out, self.combined_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.combined_hidden_size, self.combined_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.combined_hidden_size, self.combined_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.combined_hidden_size, 14)
        )

    def forward(self, image, joint_states):
        image_out = self.image_processing(image)

        # print(type(joint_states))
        # print(joint_states.dtype)
        # print(self.joint_state_processing)

        joint_states = joint_states.to(torch.float)
        


        joint_states_out = self.joint_state_processing(joint_states)


        # print(image_out.size())
        # print(joint_states_out.size())

        combined = torch.cat((image_out, joint_states_out), dim=1)

        combined_out = self.combine(combined)


        return combined_out
    
class NullspaceModel_Norm(torch.nn.Module):
    """
    Model used to predict nullspace configuration and path planning with batch normalization
    """
    def __init__(self):
        super().__init__()


        ## image processing layers
        self.num_channels1 = 10
        self.num_channels2 = 20
        self.num_channels3 = 20
        self.kernel1_size = 5
        self.kernel2_size = 5
        self.kernel3_size = 3

        self.max_pool_size = 8
        self.image_lin_out_size = 150

        self.conv1_out_size = (200 - (self.kernel1_size-1))
        self.conv2_out_size = (self.conv1_out_size - (self.kernel2_size-1) )
        self.conv3_out_size = (self.conv2_out_size - self.kernel3_size-1)
        self.max_pool_out_size = torch.square(
            torch.floor(torch.tensor((self.conv3_out_size - (self.max_pool_size - 1)-1)/self.max_pool_size + 1)).int())
        
        self.image_processing = nn.Sequential(
            nn.Conv2d(3, self.num_channels1, self.kernel1_size),
            nn.BatchNorm2d(self.num_channels1),
            nn.ReLU(),
            nn.Conv2d(self.num_channels1, self.num_channels2, self.kernel2_size),
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels2),
            nn.Conv2d(self.num_channels2, self.num_channels3, self.kernel3_size),
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels3),
            nn.MaxPool2d(self.max_pool_size),
            nn.Flatten(),
            nn.Linear(self.max_pool_out_size.item()*self.num_channels2, self.image_lin_out_size),
            nn.BatchNorm1d(self.image_lin_out_size),
            nn.ReLU(),
            nn.Linear(self.image_lin_out_size, self.image_lin_out_size),
        )


        # joint position processing layers
        self.joint_lin1_out = 100
        self.joint_lin2_out = 100


        self.joint_state_processing = nn.Sequential(
            nn.Linear(7, self.joint_lin1_out).to(torch.float),
            nn.BatchNorm1d(self.joint_lin1_out),
            nn.ReLU().to(torch.float),
            nn.Linear(self.joint_lin1_out, self.joint_lin2_out).to(torch.float),
            nn.BatchNorm1d(self.joint_lin2_out),
            nn.ReLU(),
            nn.Linear(self.joint_lin1_out, self.joint_lin2_out).to(torch.float),
        )

        ## combined processing layers
        self.combined_hidden_size = 150

        self.combine = nn.Sequential(
            nn.Linear(self.image_lin_out_size + self.joint_lin2_out, self.combined_hidden_size),
            nn.BatchNorm1d(self.combined_hidden_size),
            nn.ReLU(),
            nn.Linear(self.combined_hidden_size, self.combined_hidden_size),
            nn.BatchNorm1d(self.combined_hidden_size),
            nn.ReLU(),
            nn.Linear(self.combined_hidden_size, self.combined_hidden_size),
            nn.BatchNorm1d(self.combined_hidden_size),
            nn.ReLU(),
            nn.Linear(self.combined_hidden_size, 14)
        )

    def forward(self, image, joint_states):
        image_out = self.image_processing(image)

        joint_states = joint_states.to(torch.float)
    
        joint_states_out = self.joint_state_processing(joint_states)

        combined = torch.cat((image_out, joint_states_out), dim=1)

        combined_out = self.combine(combined)


        return combined_out


class NullspaceOnlyModel_Norm(torch.nn.Module):
    """
    Model to predict only the nullpace configuration using batch normalization"""

    def __init__(self):
        super().__init__()

        
        ## image processing layer
        self.num_channels1 = 10
        self.num_channels2 = 20
        self.kernel1_size = 5
        self.kernel2_size = 5

        self.max_pool_size = 8
        self.image_lin_out_size = 100

        self.conv1_out_size = (200 - (self.kernel1_size-1))
        self.conv2_out_size = (self.conv1_out_size - (self.kernel2_size-1) )
        self.max_pool_out_size = torch.square(
            torch.floor(torch.tensor((self.conv2_out_size - (self.max_pool_size - 1)-1)/self.max_pool_size + 1)).int())
        
        self.image_processing = nn.Sequential(
            nn.Conv2d(3, self.num_channels1, self.kernel1_size),
            nn.BatchNorm2d(self.num_channels1),
            nn.ReLU(),
            nn.Conv2d(self.num_channels1, self.num_channels2, self.kernel2_size),
            nn.ReLU(),
            nn.BatchNorm2d(self.num_channels2),
            nn.MaxPool2d(self.max_pool_size),
            nn.Flatten(),
            nn.Linear(self.max_pool_out_size.item()*self.num_channels2, self.image_lin_out_size),
            nn.BatchNorm1d(self.image_lin_out_size),
            nn.ReLU(),
            nn.Linear(self.image_lin_out_size, self.image_lin_out_size),
        )

        ## joint state processing layers
        self.joint_lin1_out = 100
        self.joint_lin2_out = 100


        self.joint_state_processing = nn.Sequential(
            nn.Linear(7, self.joint_lin1_out).to(torch.float),
            nn.BatchNorm1d(self.joint_lin1_out),
            nn.ReLU().to(torch.float),
            nn.Linear(self.joint_lin1_out, self.joint_lin2_out).to(torch.float),
            nn.BatchNorm1d(self.joint_lin2_out),
            nn.ReLU(),
            nn.Linear(self.joint_lin1_out, self.joint_lin2_out).to(torch.float),
        )

        ## combined layers
        self.combined_hidden_size = 100

        self.combine = nn.Sequential(
            nn.Linear(self.image_lin_out_size + self.joint_lin2_out, self.combined_hidden_size),
            nn.BatchNorm1d(self.combined_hidden_size),
            nn.ReLU(),
            nn.Linear(self.combined_hidden_size, self.combined_hidden_size),
            nn.BatchNorm1d(self.combined_hidden_size),
            nn.ReLU(),
            nn.Linear(self.combined_hidden_size, self.combined_hidden_size),
            nn.BatchNorm1d(self.combined_hidden_size),
            nn.ReLU(),
            nn.Linear(self.combined_hidden_size, 7)
        )

    def forward(self, image, joint_states):
        image_out = self.image_processing(image)

        joint_states = joint_states.to(torch.float)
    
        joint_states_out = self.joint_state_processing(joint_states)

        combined = torch.cat((image_out, joint_states_out), dim=1)

        combined_out = self.combine(combined)


        return combined_out
    
class NullspaceOnlyModel_Dropout(torch.nn.Module):
    """
    Model to predict only null space model using dropout regularization
    """    
    def __init__(self):
        super().__init__()

        ## set droput params
        self.dropout = 0.15
        self.dropout2d = 0.05


        ## image processing layers
        self.num_channels1 = 10
        self.num_channels2 = 20
        self.kernel1_size = 5
        self.kernel2_size = 5

        self.max_pool_size = 8
        self.image_lin_out_size = 150

        self.conv1_out_size = (200 - (self.kernel1_size-1))
        self.conv2_out_size = (self.conv1_out_size - (self.kernel2_size-1) )
        self.max_pool_out_size = torch.square(
            torch.floor(torch.tensor((self.conv2_out_size - (self.max_pool_size - 1)-1)/self.max_pool_size + 1)).int())
        
        self.image_processing = nn.Sequential(
            nn.Conv2d(3, self.num_channels1, self.kernel1_size),
            nn.ReLU(),
            nn.Dropout2d(self.dropout2d),
            nn.Conv2d(self.num_channels1, self.num_channels2, self.kernel2_size),
            nn.ReLU(),
            nn.Dropout2d(self.dropout2d),
            nn.MaxPool2d(self.max_pool_size),
            nn.Flatten(),
            nn.Linear(self.max_pool_out_size.item()*self.num_channels2, self.image_lin_out_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.image_lin_out_size, self.image_lin_out_size),
        )

        ## joint state processing layers
        self.joint_lin1_out = 100
        self.joint_lin2_out = 100

        self.joint_state_processing = nn.Sequential(
            nn.Linear(7, self.joint_lin1_out).to(torch.float),
            nn.ReLU().to(torch.float),
            nn.Dropout(self.dropout),
            nn.Linear(self.joint_lin1_out, self.joint_lin2_out).to(torch.float),
            nn.ReLU().to(torch.float),
            nn.Dropout(self.dropout),
            nn.Linear(self.joint_lin1_out, self.joint_lin2_out).to(torch.float),
        )

        ## combined layers
        self.combined_hidden_size = 100

        self.combine = nn.Sequential(
            nn.Linear(self.image_lin_out_size + self.joint_lin2_out, self.combined_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.combined_hidden_size, self.combined_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.combined_hidden_size, self.combined_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.combined_hidden_size, 7)
        )

    def forward(self, image, joint_states):
        image_out = self.image_processing(image)

        joint_states = joint_states.to(torch.float)
        
        joint_states_out = self.joint_state_processing(joint_states)

        combined = torch.cat((image_out, joint_states_out), dim=1)

        combined_out = self.combine(combined)

        return combined_out
    
