#!/usr/bin/python3
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
from timeit import default_timer as now

tf.keras.backend.set_image_data_format('channels_last')

# Only use 1 neuron core.
os.environ['NEURONCORE_GROUP_SIZES'] = '1'
print("NEURONCORE_GROUP_SIZES (env): " + os.environ.get('NEURONCORE_GROUP_SIZES', '<unset>'))

# Create input from image
img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
img_arr = image.img_to_array(img_sgl)
img_arr2 = np.expand_dims(img_arr, axis=0)
img_arr3 = resnet50.preprocess_input(img_arr2)
print(img_arr3.shape)

# Load model
start = now()
COMPILED_MODEL_DIR = './resnet50_neuron/'
predictor_inferentia = tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR)
end = now()
print('Loads model: {:.3f} ms'.format((end-start)*1000.0))

# Run inference
for i in range(1, 10):

  start = now()
  model_feed_dict={'input': img_arr3}
  infa_rslts = predictor_inferentia(model_feed_dict)
  end = now()
  print('Infer time: {:.3f} ms'.format((end-start)*1000.0))
  print(infa_rslts["output"].shape)
  # Display results
  print(resnet50.decode_predictions(infa_rslts["output"], top=5)[0])

  time.sleep(3)

