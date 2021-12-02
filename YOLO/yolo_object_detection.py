# This script is used for loading in the trained YOLO network and apply it to images it has not seen before to validate the functionality of YOLO
# @!@!@!learn how to store the get_iou function in another file to make this file smaller

import cv2
import numpy as np
import glob
import random

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights",
                      "yolov3_testing.cfg")  # load the weights(trained model which can be found in google drive)

# Name custom object (box)
classes = [""]

# Images path
images_path = glob.glob(
    './YOLO_Images_Noisy/*.jpg')  # images containing the box with a background containing lab images or noisy
groundtruth_path = glob.glob(
    './YOLO_Images_Noisy/*.txt')  # text file containing the parameters of the groundtruth box ('class','x_cent','y_cent','width','height')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))  # pick random color
with open('iou_values.csv',
          'w') as f:  # create a file csv which contains the picture number with the corresponding Intersection over Union score
    f.write('picture number;iou score')


def get_iou(bb1, bb2):  # function which gives the iou of 2 bounding boxes
    """                     #https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation#comment85551557_42874377
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# Insert here the path of your images
random.shuffle(images_path)  # start with random image from the dir: Images Noisy
# loop through all the images
for img_path in images_path:
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
            if confidence > 0.3:
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

            cv2.rectangle(img, (int(bb_groundtruth['x1'] * width), int(bb_groundtruth['y1'] * width)),
                          (int(bb_groundtruth['x2'] * width), int(bb_groundtruth['y2'] * width)), (255, 0, 0),
                          2)  # draw the groundtruth box on the image
            cv2.putText(img, str("%.3f" % iou), (x, y - 15), font, 3, color, 2)

    save_dir = img_path[:-3]  # rewriting save_dir to have the same image index as input
    save_dir = save_dir + "jpg"
    file_name_dir = save_dir[20:]
    save_dir = "./YOLO_Validation/" + file_name_dir
    iou_string = [" ", file_name_dir[:-4] + ";" + str("%.3f" % iou)]
    with open('iou_values.csv', 'a') as g:  # add the image index and iou to the already existing csv file
        g.writelines('\n'.join(iou_string))
    cv2.imwrite(save_dir, img)  # save the image in the folder:YOLO validation
    # cv2.imshow("Image", img)  # show the image (optional)
    # key = cv2.waitKey(0)      # click image away (optional)

cv2.destroyAllWindows()
