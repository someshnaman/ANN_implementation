import os
import tensorflow as tf
import numpy as np
import time
from .config_reader import Config_reader


class Callbacks:
    def __init__(self):
        pass

    def get_time_stamp(self, filemname):
        timstamp = time.asctime().replace(" ", "_").replace(":", '_')
        unique_name = f'{filemname}_at_{timstamp}'
        return unique_name

    def get_callbacks(self, X_train, content):
        logs = content['logs']
        unique_dir_name = self.get_time_stamp('tb_logs')
        tensorboard_root_log_dir = os.path.join(logs['logs_dir'], logs['TENSORBOARD_ROOT_LOG_DIR'], unique_dir_name)
        os.makedirs(tensorboard_root_log_dir, exist_ok=True)
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_root_log_dir)
        file_writer = tf.summary.create_file_writer(logdir=tensorboard_root_log_dir)
        with file_writer.as_default():
            images = np.reshape(X_train[10:30], (
            -1, 28, 28, 1))  # here I've taken 20 images and reshaping it to 1 color nchanger .-1 represents 20
            tf.summary.image("20 handwritten digit samples", images, max_outputs=25, step=0)

        params = content['params']
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=params["patience"],
            restore_best_weights=params["restore_best_weights"])
        artifacts = content['artifacts']
        CKPT_dir = os.path.join(artifacts["artifacts_dir"], artifacts["CHECKPOINT_DIR"])
        os.makedirs(CKPT_dir, exist_ok=True)
        CKPT_path = os.path.join(CKPT_dir, "model_ckpt.h5")

        checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)

        return [tensorboard_cb, early_stopping_callback, checkpointing_cb]
