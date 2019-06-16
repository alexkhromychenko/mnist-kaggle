import csv
import os
import numpy as np
from tensorflow import keras

train_image_dir = os.path.join('data', 'image', 'train')

train_csv_path = os.path.join('data', 'csv', 'train.csv')
with open(train_csv_path) as image_csv:
    reader = csv.reader(image_csv)
    next(reader)
    for index, row in enumerate(reader):
        label = row[0]
        image_pixels = np.array(row[1:]).reshape((28, 28, 1))

        pil_image = keras.preprocessing.image.array_to_img(image_pixels)
        image_path = os.path.join(train_image_dir, label, 'image_' + str(index) + '.png')
        pil_image.save(image_path)
