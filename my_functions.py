import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image/255
    return image.numpy()

def predict(image_path, model_path, labels_path, top_k=5):
    # load labels
    with open(labels_path, 'r') as labels_file:
        class_names = json.load(labels_file)
    
    # load the model
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    
    # load and process the image
    image = np.asarray(Image.open(image_path))
    image = process_image(image)
    image = tf.convert_to_tensor(image)
    image = tf.reshape(image, (1, 224, 224, 3))
    
    # predict
    probabilities = model.predict(image)
    predictions = zip(list(probabilities.squeeze()), list(range(1, 103)))
    tmp = sorted(list(predictions), key=lambda x: x[0], reverse=True)[:top_k]
    probabilities, classes = list(zip(*tmp))
    return probabilities, [class_names[str(i)] for i in classes]
