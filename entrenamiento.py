import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
dataset_path=r"C:\Users\FRANCISCO\Desktop\RNN\dataset"
IMG_SIZE=(128,128)
BATCH_SIZE=16
EPOCHS=40

datagen=ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_gen=datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True)

val_gen=datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')

model=Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)), MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'), MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'), MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation='relu'), MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.summary()
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True)

history=model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
    )

train_acc=history.history['accuracy'][-1]
val_acc=history.history['val_accuracy'][-1]
train_loss=history.history['loss'][-1]
val_loss=history.history['val_loss'][-1]
print(f"prec. entrenamiento: {train_acc*100:.2f}%")
print(f"prec. validacion: {val_acc*100:.2f}%")
print(f"error entrenamiento: {train_loss:.4f}")
print(f"error validacion: {val_loss:.4f}")
print(f"margen d error d validacion: {(1 - val_acc)*100:.2f}%")
model.save(r"C:\Users\FRANCISCO\Desktop\RNN\daniyyo.keras")
print("\nya")
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='entrenamiento')
plt.plot(history.history['val_accuracy'], label='validacion')
plt.title('precision durante el entrenamiento')
plt.xlabel('epocas')
plt.ylabel('accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='entrenamiento')
plt.plot(history.history['val_loss'], label='validacion')
plt.title('error durante el entrenamiento')
plt.xlabel('epocas')
plt.ylabel('loss')
plt.legend()
plt.show()