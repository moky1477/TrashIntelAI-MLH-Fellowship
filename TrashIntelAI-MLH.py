import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dropout
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from sklearn.metrics import classification_report

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.summary()

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
preds = Dense(7, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)

len(model.layers)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, fill_mode='nearest', horizontal_flip=True, shear_range=0.2, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

from imutils import paths
train_dir = '/content/drive/MyDrive/Small-DS-Sample/Test/train'
validation_dir = '/content/drive/MyDrive/Small-DS-Sample/Test/val'
test_dir = '/content/drive/MyDrive/Small-DS-Sample/Test/test'
totalTrain = len(list(paths.list_images(train_dir)))
totalVal = len(list(paths.list_images(validation_dir)))
totalTest = len(list(paths.list_images(test_dir)))
print("Total Training: ", totalTrain)
print("Total Validation: ", totalVal)
print("Total test: ", totalTest)

BATCH_SIZE = 16
TARGET_SIZE = (224, 224)

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=BATCH_SIZE, class_mode='categorical', color_mode='rgb', shuffle=True, target_size=TARGET_SIZE)
validation_generator = test_datagen.flow_from_directory(validation_dir, batch_size=BATCH_SIZE, color_mode='rgb', class_mode='categorical', shuffle=False, target_size=TARGET_SIZE)
test_generator = test_datagen.flow_from_directory(test_dir, class_mode="categorical", color_mode='rgb', target_size=TARGET_SIZE, shuffle=False, batch_size=BATCH_SIZE)

print("[INFO] compiling model...")
opt = Adam(learning_rate=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print("[INFO] training head...")
H = model.fit(train_generator, steps_per_epoch=totalTrain // BATCH_SIZE, validation_data=validation_generator, validation_steps=totalVal // BATCH_SIZE, epochs=10)

train_loss = H.history["loss"]
val_loss = H.history["val_loss"]
epochs = range(1, len(train_loss) + 1)

plt.style.use("ggplot")
plt.figure()
plt.plot(epochs, train_loss, label="train_loss")
plt.plot(epochs, val_loss, label="val_loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()

train_acc = H.history["accuracy"]
val_acc = H.history["val_accuracy"]

plt.figure()
plt.plot(epochs, train_acc, label="train_acc")
plt.plot(epochs, val_acc, label="val_acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()

print("[INFO] evaluating after fine-tuning network head...")
test_generator.reset()
predIdxs = model.predict(test_generator, steps=(totalTest // BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(test_generator.classes, predIdxs, target_names=test_generator.class_indices.keys()))

from tensorflow.keras.models import load_model
print("[INFO] serializing network...")
model.save('/content/drive/MyDrive/Saved Models/garbage_model-1.h5')

from keras.models import load_model
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img

import random

nrows = 8
ncols = 4
pic_index = 0
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

waste_types = ['paper', 'glass', 'clothes', 'organic', 'e-waste', 'metal', 'plastic']

test_d = '/content/drive/MyDrive/Small-DS-Sample/Test/test/' + waste_types[6] + '/'
test_files = os.listdir(test_d)
test_files = random.sample(test_files, 10)

for i, fn in enumerate(test_files):
    sp = plt.subplot(nrows, ncols, i + 1, facecolor='red')
    sp.axis('Off')
    path = test_d + fn
    image = cv2.imread(path)
    img = load_img(path, target_size=TARGET_SIZE)
    output = image.copy()
    output = imutils.resize(output, width=400)
    img = img_to_array(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, TARGET_SIZE)
    image = image.astype("float32") / 255.
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    preds = model.predict(img)[0]
    i = np.argmax(preds)
    label = waste_types[i]
    text = "{}: {:.2f}%".format(label, preds[i] * 100)
    cv2.putText(output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 0, 0), thickness=3)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), interpolation='bicubic')
plt.show()

import tensorflow as tf
import keras

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("/content/drive/MyDrive/Saved Models/garbage_model.tflite", "wb").write(tflite_model)
