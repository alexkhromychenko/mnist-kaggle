import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# from layers import BatchNormalization
# layers.BatchNormalization = BatchNormalization

IMG_SIZE = 75
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights=None)
base_model.trainable = True

# layer_after_mixed7_index = None
# for index, layer in enumerate(base_model.layers):
#     if layer.name == 'mixed7':
#         layer_after_mixed7_index = index + 1
#         break
#
# for layer in base_model.layers[:layer_after_mixed7_index]:
#     layer.trainable = False
#
# for layer in base_model.layers[layer_after_mixed7_index:]:
#     layer.trainable = True

global_average_pooling_layer = layers.GlobalAveragePooling2D()
dense_layer = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))
prediction_layer = layers.Dense(10, activation='softmax')

model = keras.Sequential([
    base_model,
    global_average_pooling_layer,
    dense_layer,
    prediction_layer
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=30,
                                   zoom_range=0.1)

validation_datagen = ImageDataGenerator(rescale=1/255)

BATCH_SIZE = 64

image_dir = os.path.join('data', 'image')
train_dir = os.path.join(image_dir, 'train')
validation_dir = os.path.join(image_dir, 'validation')
transformed_dir = os.path.join(image_dir, 'transformed')

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    target_size=(IMG_SIZE, IMG_SIZE))

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=BATCH_SIZE,
                                                              class_mode='categorical',
                                                              target_size=(IMG_SIZE, IMG_SIZE))

checkpoint_dir = os.path.join('model')
checkpoint_path = os.path.join(checkpoint_dir, 'cp.ckpt')
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True)

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest_checkpoint)

history = model.fit_generator(train_generator, epochs=10, validation_data=validation_generator, callbacks=[cp_callback])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
