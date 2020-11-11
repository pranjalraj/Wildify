# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:16:20 2020

@author: Antony J K
"""
import tensorflow as tf
import pathlib
from tensorflow.keras import layers
from tensorflow.keras import regularizers


data_dir = pathlib.Path("D:/Personal/Vtop/Semester 5/Artificial Intellegance/J Comp/raw-img")
image_count = len(list(data_dir.glob('*/*.*')))
batch_size = 8
img_height = 100
img_width = 100
train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                               validation_split=0.2,
                                                               subset="training",
                                                               seed=7,
                                                               image_size=(img_height, img_width),
                                                               batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                             validation_split=0.2,
                                                             subset="validation",
                                                             seed=7,
                                                             image_size=(img_height, img_width),
                                                             batch_size=batch_size)
class_names = train_ds.class_names
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
num_classes = len(class_names)

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(padding='same'),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(padding='same'),
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(padding='same'),
  layers.Flatten(),
  layers.Dropout(0.6),
  layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
  layers.Dropout(0.55),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=43
)
model.save("D:/Personal/Vtop/Semester 5/Artificial Intellegance/J Comp/Model/Test1")