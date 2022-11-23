#!/usr/bin/env python
# coding: utf-8

# This notebook follows instructions from mlzoomcamp 2022 and is admitted as capstone project: https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/08-deep-learning/notebook.ipynb
# 
# Problem statement: There are different entities of skin cancer. The HAM10000 is a collection of skin lesion images. Identifying the type of skin lesion is challenging. This project applies neural network techniques to classify the images. 
# 
# ***Disclaimer: This work is part of an educational project. It is not intended for clinical application. As such it can not make real world predictions for skin lesions. To get recommendations regarding skin lesions one should ask for expert advice such as provided by a dermatologist.***

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator

metadata = pd.read_csv("../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

df_img = pd.DataFrame()
df_img["img_file"] = [img_name+".jpg" for img_name in metadata.image_id]
df_img["dx"] = list(metadata.dx)

train_ds = train_gen.flow_from_dataframe(
    df_img,
    directory="../input/skin-cancer-mnist-ham10000/HAM10000_images_part_1",
    x_col="img_file",
    y_col="dx",
    target_size=(150, 150),
    batch_size=32
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_dataframe(
    df_img,
    directory="../input/skin-cancer-mnist-ham10000/HAM10000_images_part_2",
    x_col="img_file",
    y_col="dx",
    target_size=(150, 150),
    batch_size=32
)

base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

base_model.trainable = False
inputs = keras.Input(shape=(150, 150, 3))
base = base_model(inputs, training=False)
vectors = keras.layers.GlobalAveragePooling2D()(base)
outputs = keras.layers.Dense(7)(vectors)
model = keras.Model(inputs, outputs)

learning_rate = 0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v4_1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[checkpoint])