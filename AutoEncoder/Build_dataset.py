# This script gathers the images needed for training and testing and saves them in a numpy file type so they can be loaded into Google Colab
# The ratio of Train/Test should be around 0.8

import numpy as np
import cv2
import os

DATADIR_TRAIN_CLEAN = "D:\\Internship_IST\\CodePoseEstimation6D\\AutoEncoder\\Images_dataset\\train\\clean"
DATADIR_TRAIN_NOISY = 'D:\\Internship_IST\\CodePoseEstimation6D\\AutoEncoder\\Images_dataset\\train\\noisy'
DATADIR_TEST_CLEAN = "D:\\Internship_IST\\CodePoseEstimation6D\\AutoEncoder\\Images_dataset\\test\\clean"
DATADIR_TEST_NOISY = "D:\\Internship_IST\\CodePoseEstimation6D\\AutoEncoder\\Images_dataset\\test\\noisy"

training_data_clean = []
training_data_noisy = []
testing_data_clean = []
testing_data_noisy = []


def create_training_data_clean():
    path = os.path.join(DATADIR_TRAIN_CLEAN)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        training_data_clean.append(img_array)

    dataset_clean = np.array(training_data_clean)
    return dataset_clean


def create_training_data_noisy():
    path = os.path.join(DATADIR_TRAIN_NOISY)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        training_data_noisy.append(img_array)

    dataset_noisy = np.array(training_data_noisy)
    return dataset_noisy


def create_test_data_clean():
    path = os.path.join(DATADIR_TEST_CLEAN)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        testing_data_clean.append(img_array)

    dataset_clean = np.array(testing_data_clean)
    return dataset_clean


def create_test_data_noisy():
    path = os.path.join(DATADIR_TEST_NOISY)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        testing_data_noisy.append(img_array)

    dataset_noisy = np.array(testing_data_noisy)
    return dataset_noisy


train_data_clean = create_training_data_clean()
train_data_noisy = create_training_data_noisy()
test_data_clean = create_test_data_clean()
test_data_noisy = create_test_data_noisy()

np.save('train_data_clean.npy', train_data_clean)
np.save('train_data_noisy.npy', train_data_noisy)
np.save('test_data_clean.npy', test_data_clean)
np.save('test_data_noisy.npy', test_data_noisy)
