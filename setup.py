
from setuptools import setup


setup(
    name="src",
    version="0.0.2",
    author="Somesh",
    description="ANN Implementation of mnist Data.",
    long_description_content_type="text/markdown",
    url="gh repo clone someshnaman/-ANN-implementation-handwritter_mnist",
    author_email="namanayo5@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas"
    ]
)