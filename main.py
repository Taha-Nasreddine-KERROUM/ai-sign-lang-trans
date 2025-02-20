import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import tensorflow as tf

print("TensorFlow version: ", tf.__version__)

print("GPUs detected: ", tf.config.list_physical_devices('GPU'))