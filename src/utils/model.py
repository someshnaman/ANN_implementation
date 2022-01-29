import tensorflow as tf
from tensorflow.keras import layers
import time
import os



class Model:
    def __init__(self):
        pass

    def create_model(self, LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):
        self.loss_function = LOSS_FUNCTION
        self.optimizer = OPTIMIZER
        self.metrics = METRICS
        self.num_classes = NUM_CLASSES
        layers = [tf.keras.layers.Flatten(input_shape=[28, 28], name='input_layer'),
                  tf.keras.layers.Dense(300, activation='relu', name='hidden_layer1'),
                  tf.keras.layers.Dense(300, activation='relu', name='hidden_layer2'),
                  tf.keras.layers.Dense(self.num_classes, activation="softmax", name="outputLayer")]
        model_clf = tf.keras.models.Sequential(layers)
        model_clf.summary()
        model_clf.compile(loss=self.loss_function,
                          optimizer=self.optimizer,
                          metrics=self.metrics)
        return model_clf  # This will return untrained Model

    def get_unique_filename(self, filename):
        unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
        return unique_filename

    def save_model(self, model, model_name, model_dir):
        unique_filename = self.get_unique_filename(model_name)
        path_to_model = os.path.join(model_dir, unique_filename)
        model.save(path_to_model)



