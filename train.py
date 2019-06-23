from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import model_definition


load_last_checkpoint = True
model = model_definition.load_xception(load_last_checkpoint)
model.compile(optimizer=keras.optimizers.Adam(lr=0.00000001), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=30,
                                   zoom_range=0.15)

validation_datagen = ImageDataGenerator(rescale=1/255)

BATCH_SIZE = 128

image_dir = os.path.join('data', 'image')
train_dir = os.path.join(image_dir, 'train')
validation_dir = os.path.join(image_dir, 'validation')
transformed_dir = os.path.join(image_dir, 'transformed')

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    target_size=(model_definition.IMG_SIZE,
                                                                 model_definition.IMG_SIZE))

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=BATCH_SIZE,
                                                              class_mode='categorical',
                                                              target_size=(model_definition.IMG_SIZE,
                                                                           model_definition.IMG_SIZE))

checkpoint_dir = os.path.join('model')
checkpoint_path = os.path.join(checkpoint_dir, 'cp.ckpt')
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True)

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
