import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="centigrad",
    version="0.1.0",
    description="Autograd engine and neural network library based on numpy",
    author="Dinesh Kumar Gnanasekaran",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["centigrad"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "numpy",
        "scipy",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
