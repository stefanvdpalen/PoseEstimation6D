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
        if w > 900 or h > 900: continue
        cv2.rectangle(im_bw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(im_bw, 'Moth Detected', (x + w + 10, y + h), 0, 0.3, (0, 255, 0))
        cv2.imshow("Show", im_bw)
        cv2.waitKey()
        x_centre = (x + w / 2) / width_im
        y_centre = (y + h / 2) / height_im
        width_YOLO = x / width_im
        height_YOLO = y / height_im
        String_print = ["0 ", str(x_centre), " ", str(y_centre), " ", str(width_YOLO), " ", str(height_YOLO)]
        new_filename = fname
        new_filename = new_filename[:-3]
        filetype = "txt"
        print(new_filename)
        filename = new_filename + filetype
        f = open(filename, "w+")
        f.writelines(String_print)
        f.close()
