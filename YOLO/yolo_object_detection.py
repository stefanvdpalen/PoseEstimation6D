# This script is used for loading in the trained YOLO network and apply it to images it has not seen before to validate the functionality of YOLO
import os.path

import cv2
import numpy as np
import glob
import random
from YOLO_Intersection_over_union_calc import get_iou

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_4000.weights",
                      "yolov3_testing.cfg")  # load the weights(trained model which can be found in google drive)

# Name custom object (box)
classes = [""]
iou = 0

# Images path
images_path = glob.glob(
    './YOLO_Images_Real/*.jpg')  # images containing the box with a background containing lab images or noisy
groundtruth_path = glob.glob(
    './YOLO_Images_Real/*.txt')  # text file containing the parameters of the groundtruth box ('class','x_cent','y_cent','width','height')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))  # pick random color
with open('iou_values.csv',
          'w') as f:  # create a file csv which contains the picture number with the corresponding Intersection over Union score
    f.write('picture number;iou score')


# Insert here the path of your images
random.shuffle(images_path)  # start with random image from the dir: Images Noisy
# loop through all the images
for img_path in images_path:
    iou = 0
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.0:
                print(confidence)
                # Object detected
                # print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            boundingboxfile = img_path[:-3] + "txt"
            bb_estimation = {'x1': (x + 1) / width, 'x2': (x + 1 + w) / width, 'y1': (y + 1) / height,
                             'y2': (y + 1 + h) / height}

            if os.path.exists(boundingboxfile):
                with open(boundingboxfile,
                          "r") as boundboxfile:  # open the text file which contains the groundtruth value of the box
                    boundbox_values = boundboxfile.readlines()
                _, x_cent_bb1, y_cent_bb1, width_bb1, height_bb1 = boundbox_values[0].split(" ")
                bb_groundtruth = {'x1': float(x_cent_bb1) - 1 / 2 * float(width_bb1),
                                  'x2': float(x_cent_bb1) + 1 / 2 * float(width_bb1),
                                  'y1': float(y_cent_bb1) - 1 / 2 * float(height_bb1),
                                  'y2': float(y_cent_bb1) + 1 / 2 * float(
                                      height_bb1)}  # rewrite the values for the get_iou function which has as input top left en bottom right coordinates
                iou = get_iou(bb_estimation, bb_groundtruth)

                cv2.rectangle(img, (int(bb_groundtruth['x1'] * width), int(bb_groundtruth['y1'] * height)),
                              (int(bb_groundtruth['x2'] * width), int(bb_groundtruth['y2'] * height)), (255, 0, 0),
                              2)  # draw the groundtruth box on the image
                cv2.putText(img, str("%.3f" % iou), (x, y - 15), font, 3, color, 2)

    save_dir = img_path[:-3]  # rewriting save_dir to have the same image index as input
    save_dir = save_dir + "jpg"
    file_name_dir = save_dir[19:]
    save_dir = "./YOLO_Validation/" + file_name_dir
    iou_string = [" ", file_name_dir[:-4] + ";" + str("%.3f" % iou)]
    print(save_dir)
    with open('iou_values.csv', 'a') as g:  # add the image index and iou to the already existing csv file
        g.writelines('\n'.join(iou_string))
    cv2.imwrite(save_dir, img)  # save the image in the folder:YOLO validation
    print("image saved")
    #cv2.imshow("Image", img)  # show the image (optional)
    key = cv2.waitKey(0)      # click image away (optional)

cv2.destroyAllWindows()
