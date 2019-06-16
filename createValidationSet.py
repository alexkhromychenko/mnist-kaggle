import os

image_dir = os.path.join('data', 'image')
train_image_dir = os.path.join(image_dir, 'train')
validation_image_dir = os.path.join(image_dir, 'validation')

for number_int in range(10):
    number = str(number_int)
    number_train_image_dir = os.path.join(train_image_dir, number)
    number_validation_image_dir = os.path.join(validation_image_dir, number)

    images_to_move = os.listdir(number_train_image_dir)[:100]
    for image in images_to_move:
        old_path = os.path.join(number_train_image_dir, image)
        new_path = os.path.join(number_validation_image_dir, image)
        os.rename(old_path, new_path)
