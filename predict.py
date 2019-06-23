import csv
import os
import numpy as np
from tensorflow import keras
import model_definition

IMG_SIZE = model_definition.IMG_SIZE
INPUT_SHAPE = (0, IMG_SIZE, IMG_SIZE, 3)

BATCH_SIZE = 128

csv_dir = os.path.join('data', 'csv')
test_csv_path = os.path.join(csv_dir, 'test.csv')

model = model_definition.load_xception()

input_img_arrays = np.empty(INPUT_SHAPE)
predictions = []
with open(test_csv_path) as image_csv:
    reader = csv.reader(image_csv)
    next(reader)
    for index, row in enumerate(reader):
        image_pixels = np.array(row[:]).reshape((28, 28, 1))

        pil_image = keras.preprocessing.image.array_to_img(image_pixels)
        pil_image = pil_image.resize((IMG_SIZE, IMG_SIZE))

        img_array = keras.preprocessing.image.img_to_array(pil_image)
        img_array = np.repeat(img_array, 3, axis=2)
        img_array = img_array / 255
        img_array = np.expand_dims(img_array, axis=0)

        input_img_arrays = np.append(input_img_arrays, img_array, axis=0)

        if (index + 1) % BATCH_SIZE == 0:
            result = model.predict_classes(input_img_arrays)
            predictions.extend(result)

            input_img_arrays = np.empty(INPUT_SHAPE)

if len(input_img_arrays) > 0:
    result = model.predict_classes(input_img_arrays)
    predictions.extend(result)

result_csv_path = os.path.join(csv_dir, 'result.csv')
with open(result_csv_path, mode='w', newline='') as result_csv:
    result_writer = csv.writer(result_csv)
    result_writer.writerow(['ImageId', 'Label'])
    for index, prediction in enumerate(predictions):
        result_writer.writerow([index + 1, prediction])
