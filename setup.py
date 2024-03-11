from setuptools import find_packages, setup

setup(
    name="sentinel2_ts",
    author="Anthony Frion",
    author_email="anthony.frion@imt-atlantique.fr",
    python_requires=">=3.11",
    url="https://github.com/anthony-frion/Sentinel2TS",
    install_requires=[
        "torch==2.2.1",
        "torchvision==0.17.1",
        "pytorch-lightning==2.1.3",
        "scikit-learn==1.3.0",
        "numpy==1.24.3",
        "matplotlib==3.8.0",
    ],
    packages=find_packages(),
)
