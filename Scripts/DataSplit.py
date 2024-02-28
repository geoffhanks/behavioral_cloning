from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import os

import csv



"""
Split dataset into training, validation and testing sets
"""
def main():

    # path to the data set
    data_dir = "/home/airlab4/ros_ws/src/shared_control/data/nullspace_and_stiffness_BC/dataset9"

    # data file path
    data_file_path = os.path.join(data_dir, "data.csv")

    # read data file into dataframe
    data_df = pd.read_csv(data_file_path, header=None)

    # get the indeces of each data point
    data_indeces = np.array(data_df.iloc[:, 0].values)

    # split into training and testing sets
    train_indeces, test_indeces = train_test_split(data_indeces, train_size=0.75)
    
    # split test set into testing and validation sets
    test_indeces, val_indeces = train_test_split(test_indeces, train_size=0.5)

    # display dataset statistics
    print(f"Training set size: {train_indeces.shape[0]}")
    print(f"Testing set size: {test_indeces.shape[0]}")
    print(f"Validation set size: {val_indeces.shape[0]}")

    # save the new data files
    data_df.iloc[train_indeces, :].to_csv("./Data/nullspace_and_stiffness_BC/trainfile9.csv", sep=',', header=None)
    data_df.iloc[test_indeces, :].to_csv("./Data/nullspace_and_stiffness_BC/testfile9.csv", sep=',', header=None)
    data_df.iloc[val_indeces, :].to_csv("./Data/nullspace_and_stiffness_BC/valfile9.csv", sep=',', header=None)

if __name__ == "__main__":
    main()