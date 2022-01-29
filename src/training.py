from src.utils.data_mgmt import get_data
from src.utils.model import Model
from src.utils.config_reader import Config_reader
from src.utils.callbacks import Callbacks
import os
import argparse


def training(config_path):
    content = Config_reader(config_path)
    # Getting the data
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(config_path)
    LOSS_FUNCTION = content["params"]["loss_function"]
    OPTIMIZER = content["params"]["optimizer"]
    METRICS = content["params"]["metrics"]
    NUM_CLASSES = content["params"]["num_classes"]
    # creating the model
    model = Model()
    model = model.create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)
    EPHOCH = content["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)

    # creating callbacks
    callbacks = Callbacks()
    callback_lists = callbacks.get_callbacks(X_train, content)
    history = model.fit(X_train, y_train, epochs=EPHOCH,
                        validation_data=VALIDATION_SET, callbacks=callback_lists)
    artifacts_dir = content["artifacts"]["artifacts_dir"]
    model_dir = content["artifacts"]["model_dir"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    model_name = content["artifacts"]["model_name"]
    saving_the_model = Model()
    saving_the_model.save_model(model, model_name, model_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="../config.yaml")


    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)
