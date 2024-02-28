#!/usr/bin/env python3
import torch
from torch import nn


"""
    Requires a dictionary of parameters with the following keys:
        - im_conv_channels: list of the number of channels for each convolutional layer.  length needs to match that of im_conv_kernel_sizes
        - im_conv_kernel_sizes: list of the kernel sizes for each convolutional layer. im_conv_channels
        - im_maxpool_kenel_sizes: list of kernel sizes for the max pooling layer(s)
        - im_lin_sizes: list with the size of the linear layers that follow the image convolution
        - lin_sizes: list with the sizes of the linear layers for the non image inputs
        - cmb_sizes: list with the sizes of the linear layers for the combined image and non image inputs.  Last size should be desired number of outputs
        - dropout: value of the dropout rate
        - im_conv_dropout: dropout rate for conv2d layers
    """    
class NullspaceModelConfigurable(torch.nn.Module):

    def __init__(self, paramDict):
        super().__init__()

        self.key_names = ["im_conv_channels", "im_conv_kernel_sizes", "im_maxpool_kenel_sizes", 
                          "im_lin_sizes", "lin_sizes", "cmb_sizes", 
                          "lin_dropout", "im_conv_dropout"]

        if not all(name in paramDict for name in self.key_names):
            missingKeys = set(self.key_names) - paramDict.keys()
            raise Exception(f"Missing the following keys: {missingKeys}")
        
   
        

        self.image_processing = nn.Sequential()


        ## generate the convolution portion of image processing
        num_conv = len(paramDict["im_conv_channels"])

        output_channels = 3
        im_output_size = 200


        for i in range(num_conv - 1):
            input_channels = output_channels
            output_channels = self.im_conv_channels[i]
            kernel_size = self.im_conv_kernel_sizes[i]

            im_output_size = im_output_size - (kernel_size - 1)

            self.image_processing.add_module(f"conv{i}", nn.Conv2d(input_channels, output_channels,  kernel_size))
            self.image_processing.add_module(f"relu{i}", nn.ReLU())
            self.image_processing.add_module(f"dropout{i}", nn.Dropout2d(self.im_conv_dropout))


        ## create max pool layers
        
        num_maxpool = len(self.im_maxpool_kernel_sizes)

        for i in range(num_maxpool):
            kernel_size = self.max_pool_size
            im_output_size = torch.floor(
                torch.tensor((im_output_size - (kernel_size - 1)-1)/kernel_size + 1)).int()
            
            self.image_processing.add_module(f"maxpool{i}", nn.MaxPool2d(kernel_size))


        ## add flattening layer
        self.image_processing.add_module("flatten", nn.Flatten())


        ## create image processing linear layers

        im_output_size = torch.square(im_output_size)
        
        im_num_lin = len(self.im_lin_sizes)


        for i in range(im_num_lin):
            input_size = im_output_size
            im_output_size = self.im_lin_sizes[i]

            self.image_processing.add_module(f"lin{i}", nn.Linear(input_size, im_output_size))

            if i != im_num_lin:
                self.image_processing.add_module(f"linRelu{i}", nn.ReLU())
                self.image_processing.add_module(f"linDropout{i}", nn.Dropout(self.lin_dropout))

    
        ## creat linear layers for non image inputs

        self.lin_processing = nn.Sequential()

        num_lin = len(self.lin_sizes)

        lin_output_size = 7

        for i in range(num_lin):
            lin_input_size = lin_output_size
            lin_output_size = self.lin_sizes[i]

            self.lin_processing.add_module(f"lin{i}", nn.Linear(lin_input_size, lin_output_size))
            
            if i != num_lin-1:
                self.lin_processing.add_module(f"linRelu{i}", nn.ReLU())
                self.lin_processing.add_module(f"linDropout{i}", nn.Dropout(self.lin_dropout))


        # create combined model

        self.combined = nn.Sequential()
        
        cmb_output_size = lin_output_size + im_output_size

        num_combined = len(self.cmb_sizes)

        for i in range(num_combined):
            cmb_input_size = cmb_output_size
            cmb_output_size = self.cmb_sizes[i]

            self.lin_processing.add_module(f"lin{i}", nn.Linear(cmb_input_size, cmb_output_size))

            if (i != num_combined - 1):
                self.lin_processing.add_module(f"linRelu{i}", nn.ReLU())
                self.lin_processing.add_module(f"linDropout{i}", nn.Dropout(self.lin_dropout))

        


            
        # self.connected_input_size = torch.square(torch.floor(torch.tensor((200 - 5)/5))).int() * 5

        # self.dropout = 0.2

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
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Conv2d(self.num_channels1, self.num_channels2, self.kernel2_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.MaxPool2d(self.max_pool_size),
            nn.Dropout(self.dropout),
            nn.Flatten(),
            nn.Linear(self.max_pool_out_size.item()*self.num_channels2, self.image_lin_out_size),
        )

        self.joint_lin1_out = 50
        self.joint_lin2_out = 50


        self.joint_state_processing = nn.Sequential(
            nn.Linear(7, self.joint_lin1_out).to(torch.float),
            nn.Dropout(self.dropout),
            nn.ReLU().to(torch.float),
            nn.Linear(self.joint_lin1_out, self.joint_lin2_out).to(torch.float),
        )

        self.combined_hidden_size = 100

        self.combine = nn.Sequential(
            nn.Linear(self.image_lin_out_size + self.joint_lin2_out, self.combined_hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.combined_hidden_size, self.combined_hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
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
    
    def unpackParameters(self, paramDict):
        self.im_conv_channels = paramDict["im_conv_channels"]
        self.im_conv_kernel_sizes = paramDict["im_conv_kernel_sizes"]
        self.im_maxpool_kernel_sizes = paramDict["im_maxpool_kenel_sizes"]
        self.im_lin_sizes = paramDict["im_lin_sizes"]
        self.lin_sizes = paramDict["lin_sizes"]
        self.cmb_sizes = paramDict["cmb_sizes"]
        self.lin_dropout = paramDict["lin_dropout"]
        self.im_conv_dropout = paramDict["im_conv_dropout"]
