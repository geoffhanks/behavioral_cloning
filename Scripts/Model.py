import torch
from torch import nn



class BCModel(torch.nn.Module):
        
    def __init__(self):
        super().__init__()

        # self.connected_input_size = torch.square(torch.floor(torch.tensor((200 - 5)/5))).int() * 5

        self.num_channels1 = 10
        self.num_channels2 = 20
        self.kernel1_size = 8
        self.kernel2_size = 5

        self.max_pool_size = 8
        self.image_lin_out_size = 100

        self.conv1_out_size = (200 - (self.kernel1_size-1))
        self.conv2_out_size = (self.conv1_out_size - (self.kernel2_size-1) )
        self.max_pool_out_size = torch.square(
            torch.floor(torch.tensor((self.conv2_out_size - (self.max_pool_size - 1)-1)/self.max_pool_size + 1)).int())
        
        self.image_processing = nn.Sequential(
            nn.Conv2d(3, self.num_channels1, self.kernel1_size),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Conv2d(self.num_channels1, self.num_channels2, self.kernel2_size),
            # nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool2d(self.max_pool_size),
            nn.Flatten(),
            nn.Linear(self.max_pool_out_size.item()*self.num_channels2, self.image_lin_out_size)
        )

        self.joint_lin1_out = 50
        self.joint_lin2_out = 50


        self.joint_state_processing = nn.Sequential(
            nn.Linear(7, self.joint_lin1_out).to(torch.float),
            # nn.Dropout(),
            nn.ReLU().to(torch.float),
            nn.Linear(self.joint_lin1_out, self.joint_lin2_out).to(torch.float),
            # nn.Dropout()
        )

        self.combined_hidden_size = 100

        self.combine = nn.Sequential(
            nn.Linear(self.image_lin_out_size + self.joint_lin2_out, 100),
            nn.ReLU(),
            nn.Linear(self.combined_hidden_size, 3)
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
