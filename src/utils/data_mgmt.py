import tensorflow as tf
from .config_reader import Config_reader


def get_data(path):
    content = Config_reader(path)
    validation_number = content['params']['validation_dataset']

    mnist = tf.keras.datasets.mnist
    (X_train_Full, y_train_full), (X_test, y_test) = mnist.load_data()
    # All 60,000 data is into X_train_full, y_train_full, X_test, y_test
    # Now we'll create the validation data and train data out the X_train_full, y_train_full
    # We'll devide the data with 255 pixel values to scale it down to m0 to 1
    X_valid, X_train = X_train_Full[:validation_number]/255, X_train_Full[validation_number:]/255
    y_valid, y_train = y_train_full[:validation_number], y_train_full[validation_number:]

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

