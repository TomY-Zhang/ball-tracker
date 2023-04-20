# flake8: noqa
import os, tensorflow as tf, numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(69420)

class CocoModel:
    def __init__(self, model_url, cache_dir):
        self.download_model(model_url, cache_dir)
        self.load_model()

    def download_model(self, model_url, cache_dir):
        # parse path string to get the model name
        filename = os.path.basename(model_url)
        self.model_name = filename[:filename.find('.')]

        # create cache directory to store pretrained models
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # pull file from tensorflow model repo
        get_file(fname=filename, origin=model_url, cache_dir=self.cache_dir,
                 cache_subdir="checkpoints", extract=True)
        

    def load_model(self):
        print(f"Loading model '{self.model_name}'")

        # clear session to free up memory
        tf.keras.backend.clear_session()

        # load model from cache directory
        path_to_model = os.path.join(self.cache_dir, "checkpoints", self.model_name, "saved_model")
        self.model = tf.saved_model.load(path_to_model)

        print(f"Model '{self.model_name}' loaded successfully!")