# Sentinel2TS
The goal is to perform semantic unmixing on Sentinel2 Time Series

# Data
The data used are the same as the original repository.
You can find the numpy files corresponding to the time series [here](https://drive.google.com/drive/folders/1doHnjryCMptkzxYFfw-ILwAD0tOK3LGH?usp=sharing).


# How to run

## Requirements
Hopefully the source code should be installable as a package with all of its requirements.
Else here is a list of the requirements

    "torch==2.2.1",
    "torchvision==0.17.1",
    "pytorch-lightning==2.1.3",
    "scikit-learn==1.3.0",
    "numpy==1.24.3",
    "matplotlib==3.8.0",

## Extract a dataset
Data should be in the format `(number_time_steps, number_bands, x_range, y_range)` and saved in a file `data.npy`

Run:
```bash
python scripts/extract_dataset.py --data_path path/to/data --dataset_name name_of_dataset
```
This script will extract every time series into a single invidual npy file

## Train a model
A varitety of different models are available to train looking through 
```bash
scripts/train.py -h
```
will help

For training a koopman auto-encoder for example run:
```bash
scripts/train.py --train_data_path datasets/your_dataset --val_data_path datasets/your_dataset --experiment_name koopman_example --mode koopman_ae --batch_size 512 --max_epochs 100
```

## Evaluation
Once a model is trained depending on what model was chosen, first compute the abundance map, the endmembers and the reconstruction


Then run the corresponding evaluation to get the desired metrics


# Acknowledgement :

This repository was forked from Anthony Frion's [repository](https://github.com/anthony-frion/Sentinel2TS)

## Associated papers

Frion, A., Drumetz, L., Tochon, G., Dalla Mura, M. & Aïssa El Bey, A. (2023). Learning Sentinel-2 reflectance dynamics for data-driven assimilation and forecasting. EUSIPCO 2023. arXiv:2305.03743.

Frion, A., Drumetz, L., Mura, M. D., Tochon, G., & Bey, A. A. E. (2023). Neural Koopman prior for data assimilation. arXiv preprint arXiv:2309.05317.

File `sentinel2_ts/architectures/koopman_ae.py` contains the implementation of the Koopman auto-encoder model discussed in the papers.

