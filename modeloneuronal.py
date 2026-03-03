import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
modelo=tf.keras.models.load_model(r"C:\Users\FRANCISCO\Desktop\RNN\daniyyov2.keras")
datagen=ImageDataGenerator(rescale=1./255)
temp_gen=datagen.flow_from_directory(
    r"C:\Users\FRANCISCO\Desktop\RNN\dataset",
    target_size=(128,128),
    batch_size=1,
    class_mode='categorical',
    shuffle=False)
clases=list(temp_gen.class_indices.keys())
print("orden real de clases:", clases)
ruta_imagen=r"C:\Users\FRANCISCO\Desktop\RNN\dataset\francis_feliz\foto12.jpg"
img=image.load_img(ruta_imagen, target_size=(128,128))
img_array=image.img_to_array(img)/255.0
img_array=np.expand_dims(img_array, axis=0)
pred=modelo.predict(img_array)
indice=np.argmax(pred)
probabilidad=pred[0][indice]
print("clase predicha:", clases[indice])
print(f"confianza del modelo: {probabilidad*100:.2f}%")
print(f"margen de error estimado: {(1 - probabilidad)*100:.2f}%")
for i, prob in enumerate(pred[0]):
    print(f"{clases[i]}: {prob*100:.2f}%")