import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
DATADIR = "D:\Internship_IST\CodePoseEstimation6D\AutoEncoder\Training_images"
CATEGORIES = ["Clean","Noisy"]

training_data=[]

def create_training_data():
  for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
      img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
      training_data.append([img_array, class_num])

create_training_data()

np.save('train_image.npy', training_data)
print(len(training_data))
