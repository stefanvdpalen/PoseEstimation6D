# This script takes in a black and white image of the box to determine the bounding box of the actual object.
# It reads the file and stores the centre point and width,height of the object in a text file which is used for training the YOLO network
import cv2
import glob

images = glob.glob('./YOLO_Images_Clean/*.jpg')
for count, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    height_im, width_im = im_bw.shape
    contours, _ = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 0.9*width_im or h > 0.9*height_im: continue
        x_centre = (x + w / 2) / width_im
        y_centre = (y + h / 2) / height_im
        width_YOLO = w / width_im
        height_YOLO = h / height_im
        String_print = ["0 ", str(x_centre), " ", str(y_centre), " ", str(width_YOLO), " ", str(height_YOLO)]
        cv2.rectangle(im_bw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(im_bw, "Box detected", (x + w + 10, y + h), 0, 0.3, (0, 255, 0))
        #cv2.imshow("Show", im_bw)
        #cv2.waitKey()
        new_filename = fname
        new_filename = new_filename[:-3]
        filetype = "txt"
        print(new_filename)
        filename = new_filename + filetype
        f = open(filename, "w+")
        f.writelines(String_print)
        f.close()
