import datetime
import glob
import os

import cv2
import numpy as np
import pandas as pd

csv_file_name = glob.glob('*.csv')[0]
aug_type = 'mask'

df = pd.read_csv(csv_file_name)
augmented_data = [['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]
date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
augmented_images_directory = '%s-%s-imgs' % (date, aug_type)
os.makedirs(augmented_images_directory)

maxrange = len(df)

for i in range(0, maxrange):
    print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print("%s/%s" % (i+1, maxrange))

    filename, imgclass = str(df.iloc[i]['filename']), str(df.iloc[i]['class'])
    width, height = df.iloc[i]['width'], df.iloc[i]['height']
    xmin, xmax = df.iloc[i]['xmin'], df.iloc[i]['xmax']
    ymin, ymax = df.iloc[i]['ymin'], df.iloc[i]['ymax']
    print(filename)
    print(imgclass)
    img = cv2.imread(filename)[:, :, ::-1]

    X_DIMENSION = height
    Y_DIMENSION = width
    black_image = np.zeros((X_DIMENSION, Y_DIMENSION))

    filename, extension = os.path.splitext(filename)[0], os.path.splitext(filename)[1]
    old_filename = os.path.join(augmented_images_directory,"%s_%s%s" % (filename, imgclass, extension))
    new_filename = "%s_%s_%s.png" % (filename, imgclass, aug_type)
    new_filename_wo_extension = os.path.splitext(new_filename)[0]
    implant_filename = os.path.join(augmented_images_directory, "%s_implant.jpg" % new_filename_wo_extension)

    cv2.imwrite(old_filename, img)

    implant = img[ymin-10:ymax+10, xmin-10:xmax+10]
    cv2.imwrite(implant_filename, implant)
    implant = cv2.imread(implant_filename, 0)
    implant = cv2.equalizeHist(implant)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    implant = clahe.apply(implant)
    implant = cv2.GaussianBlur(implant, (25, 25), 0)

    black_image[ymin-10:ymax+10, xmin-10:xmax+10] = implant

    cv2.imwrite(implant_filename, implant)
    cv2.imwrite(os.path.join(augmented_images_directory, new_filename), black_image)

    img = cv2.imread(os.path.join(augmented_images_directory, new_filename), 0)

    # th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # ret3, th3 = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite(os.path.join(augmented_images_directory, new_filename), th1)

print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print("DONE!")
