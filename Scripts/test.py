
import torch
import matplotlib.pyplot as plt

from Model import BCModel
from NullspaceModel import NullspaceModel, NullspaceModel_Norm, NullspaceOnlyModel_Norm, NullspaceOnlyModel_Dropout

import pandas as pd

from torchvision.io import read_image

import tikzplotlib

def main():

    # creat model object of correct type
    model = NullspaceOnlyModel_Dropout()
    # model = NullspaceModel()

    # load desired model weights
    model.load_state_dict(torch.load("./Models/nullspace_and_stiffness_BC/Model9-8.pt"))
    # set to evaluation mode
    model.eval()

    # load the dataset to test on
    data = pd.read_csv("/home/airlab4/ros_ws/src/shared_control/data/nullspace_and_stiffness_BC/dataset9/data.csv", header=None)

    # parameters indicating the start point and span of the data plotted
    trial_num = 10
    span = 80

    # get the actual null space joint configurations
    joint1_actual = data.iloc[(trial_num*span + 1):((trial_num+1)*span), 15].values
    joint2_actual = data.iloc[(trial_num*span + 1):((trial_num+1)*span), 16].values
    joint3_actual = data.iloc[(trial_num*span + 1):((trial_num+1)*span), 17].values
    joint4_actual = data.iloc[(trial_num*span + 1):((trial_num+1)*span), 18].values
    joint5_actual = data.iloc[(trial_num*span + 1):((trial_num+1)*span), 19].values
    joint6_actual = data.iloc[(trial_num*span + 1):((trial_num+1)*span), 20].values
    joint7_actual = data.iloc[(trial_num*span + 1):((trial_num+1)*span), 21].values

    # create lists to store the predictions    
    joint1_pred = []
    joint2_pred = []
    joint3_pred = []
    joint4_pred = []
    joint5_pred = []
    joint6_pred = []
    joint7_pred = []
    index = []


    # run loop over each sample
    for i in range(trial_num*span + 1,((trial_num+1)*span)):

        # get the image associated with this dataset
        image = (read_image("/home/airlab4/ros_ws/src/shared_control/data/nullspace_and_stiffness_BC/dataset9/"  + "image" + str(i) + ".jpeg"))/255
        
        # format image
        image = torch.unsqueeze(image, 0)
        joint_states = torch.tensor(data.iloc[i, 1:8].values)
        joint_states = torch.unsqueeze(joint_states,0)
        
        # make prediction
        out =  model(image, joint_states)

        # save predictions to appropriate lists
        joint1_pred.append(out[0][0].item())
        joint2_pred.append(out[0][1].item())
        joint3_pred.append(out[0][2].item())
        joint4_pred.append(out[0][3].item())
        joint5_pred.append(out[0][4].item())
        joint6_pred.append(out[0][5].item())
        joint7_pred.append(out[0][6].item())

        index.append(i)

    # plot parameters
    subtitle_fontsize = 10  
    lengend_fontsize = 5
    axis_ticksize = 8
    line_style = "solid"

    # plot the joint predictions vs the actual values
    joint1_error = [joint1_actual[i] - joint1_pred[i] for i in range(len(index))]
    joint2_error = [joint2_actual[i] - joint2_pred[i] for i in range(len(index))]
    joint3_error = [joint3_actual[i] - joint3_pred[i] for i in range(len(index))]
    joint4_error = [joint4_actual[i] - joint4_pred[i] for i in range(len(index))]
    joint5_error = [joint5_actual[i] - joint5_pred[i] for i in range(len(index))]
    joint6_error = [joint6_actual[i] - joint6_pred[i] for i in range(len(index))]
    joint7_error = [joint7_actual[i] - joint7_pred[i] for i in range(len(index))]

    index = [index[i] - index[0] for i in range(len(index))]


    fig, ax = plt.subplots(4, 2, sharex=True)

    plt.suptitle("Predictions vs actual")
    # plt.ylabel("Loss (MSE)")
    fig.text(0.5, 0.04, 'Data Sample Index', ha='center')
    fig.text(0.04, 0.5, 'Loss (MSE)', va='center', rotation='vertical')
    
    ax[0, 0].plot(index, joint1_actual, 'k', linestyle=line_style, label="Actual")
    ax[0, 0].plot(index, joint1_pred, 'k', alpha=0.8, label="Predicted")
    ax[0, 0].set_title("Joint 1", fontsize=subtitle_fontsize)
    # ax[0, 0].set_ylabel("Loss (MSE)")
    ax[0, 0].grid()
    ax[0, 0].tick_params(axis='x', labelsize=axis_ticksize)
    ax[0, 0].tick_params(axis='y', labelsize=axis_ticksize)

    ax[0, 0].legend(fontsize=lengend_fontsize)



    ax[0, 1].plot(index, joint2_actual, 'k',linestyle=line_style, label="Actual")
    ax[0, 1].plot(index, joint2_pred, 'k', alpha=0.8, label="Predicted")
    ax[0, 1].set_title("Joint 2", fontsize=subtitle_fontsize)
    ax[0, 1].grid()
    ax[0, 1].tick_params(axis='x', labelsize=axis_ticksize)
    ax[0, 1].tick_params(axis='y', labelsize=axis_ticksize)

    ax[1, 0].plot(index, joint3_actual, 'k',linestyle=line_style, label="Actual")
    ax[1, 0].plot(index, joint3_pred, 'k', alpha=0.8, label="Predicted")
    ax[1, 0].set_title("Joint 3", fontsize=subtitle_fontsize)
    ax[1, 0].grid()
    ax[1, 0].tick_params(axis='x', labelsize=axis_ticksize)
    ax[1, 0].tick_params(axis='y', labelsize=axis_ticksize)

    ax[1, 1].plot(index, joint4_actual, 'k',linestyle=line_style, label="Actual")
    ax[1, 1].plot(index, joint4_pred, 'k', alpha=0.8, label="Predicted")
    ax[1, 1].set_title("Joint 4", fontsize=subtitle_fontsize)
    ax[1, 1].grid()
    ax[1, 1].tick_params(axis='x', labelsize=axis_ticksize)
    ax[1, 1].tick_params(axis='y', labelsize=axis_ticksize)

    ax[2, 0].plot(index, joint5_actual, 'k',linestyle=line_style, label="Actual")
    ax[2, 0].plot(index, joint5_pred, 'k', alpha=0.8, label="Predicted")
    ax[2, 0].set_title("Joint 5", fontsize=subtitle_fontsize)
    ax[2, 0].grid()
    ax[2, 0].tick_params(axis='x', labelsize=axis_ticksize)
    ax[2, 0].tick_params(axis='y', labelsize=axis_ticksize)

    ax[2, 1].plot(index, joint6_actual, 'k',linestyle=line_style, label="Actual")
    ax[2, 1].plot(index, joint6_pred, 'k', alpha=0.8, label="Predicted")
    ax[2, 1].set_title("Joint 6", fontsize=subtitle_fontsize)
    ax[2, 1].grid()
    ax[2, 1].tick_params(axis='x', labelsize=axis_ticksize)
    ax[2, 1].tick_params(axis='y', labelsize=axis_ticksize)

    ax[3, 0].plot(index, joint7_actual, 'k',linestyle=line_style, label="Actual")
    ax[3, 0].plot(index, joint7_pred, 'k', alpha=0.8, label="Predicted")
    ax[3, 0].set_title("Joint 7", fontsize=subtitle_fontsize)
    ax[3, 0].grid()
    ax[3, 0].tick_params(axis='x', labelsize=axis_ticksize)
    ax[3, 0].tick_params(axis='y', labelsize=axis_ticksize)


    ax[3, 1].plot(index, joint1_error, label="1")
    ax[3, 1].plot(index, joint2_error, label="2")
    ax[3, 1].plot(index, joint3_error, label="3")
    ax[3, 1].plot(index, joint4_error, label="4")
    ax[3, 1].plot(index, joint5_error, label="5")
    ax[3, 1].plot(index, joint6_error, label="6")
    ax[3, 1].plot(index, joint7_error, label="7")
    ax[3, 1].set_title("Joint Errors", fontsize=subtitle_fontsize)
    ax[3, 1].grid()
    ax[3, 1].tick_params(axis='x', labelsize=axis_ticksize)
    ax[3, 1].tick_params(axis='y', labelsize=axis_ticksize)

    plt.legend(fontsize=lengend_fontsize, ncol=2)

    tikzplotlib_fix_ncols(fig)

    tikzplotlib.save("./plots/predictions.tex", axis_height="\\figheight", axis_width="\\figwidth")
    
    plt.show()

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


if __name__ == "__main__":
    main()