import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt

import tikzplotlib

"""
    Plot stored training and validation curves
"""
def main():
    
    # path to the saved data
    file_path = './training_curves9-8.json'

    # read data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # create figure
    fig = plt.figure()

    index = [i for i in range(1, len(data['train_loss'])+1)]

    plt.plot(index, data['train_loss'], label='Training Loss')
    plt.plot(index, data['val_loss'], label='Validation Loss')


    plt.grid()
    plt.title("Training and Validation Losses per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()

    tikzplotlib_fix_ncols(fig)

    tikzplotlib.save("./plots/trainingLoss.tex", axis_height="\\figheight", axis_width="\\figwidth")

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