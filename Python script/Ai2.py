import sys

import random
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def training(training_batches, validating_batches):
    checkpoint_path = "D:/AI/Images/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

    mobile = tensorflow.keras.applications.mobilenet.MobileNet()

    x = mobile.layers[-6].output
    predictions = Dense(38, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=predictions)

    for layer in model.layers[:-5]:
        layer.trainable = False

    model.compile(optimizer='adam', loss=tensorflow.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    history = model.fit(x=training_batches, validation_data=validating_batches, epochs=10, callbacks=cp_callback, verbose=1)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(10)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    model.save("full_model")
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
    tflite_model1 = converter.convert()
    with open('lite1model.tflite', 'wb') as f:
        f.write(tflite_model1)


train_path = "D:/AI/Images/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
valid_path = "D:/AI/Images/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"
DS_SEED = random.randint(100, 999)

train_batches = ImageDataGenerator(
    preprocessing_function=tensorflow.keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path,
                                                                                                             target_size=(
                                                                                                                 256,
                                                                                                                 256),
                                                                                                             batch_size=64)
valid_batches = ImageDataGenerator(
    preprocessing_function=tensorflow.keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path,
                                                                                                             target_size=(
                                                                                                                 256,
                                                                                                                 256),
                                                                                                             batch_size=64)
training(train_batches, valid_batches)
