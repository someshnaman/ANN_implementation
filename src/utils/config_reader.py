import yaml


def Config_reader(path):
    """

    :type path: path of the yaml file where all your configuration details is present
    """
    with open(path) as config_file:
        content = yaml.safe_load(config_file)

    return content
