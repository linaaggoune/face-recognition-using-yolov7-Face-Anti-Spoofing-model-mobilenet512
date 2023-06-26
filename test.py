import torch
import tensorflow as tf
from tensorflow import keras
import h5py
# Load the HDF5 model file
model_file = "C:\\Users\\amgsoft\\Downloads\\vs_code_yolo\\yolov7\\mobilenetv2-epoch_80.hdf5"
h5_model = h5py.File(model_file, 'r')

# Convert the HDF5 model to a Keras model
model = keras.models.load_model(h5_model)

# Close the HDF5 file
#h5_model.close()