# XMO Spectrogram Dataset

- `create_dataset.py` - script to generate spectograms and plots from the remote catalog
- `peak_model.ipynb` - basic peak finding model using `scipy`

## Setup

```sh
conda create -n xmo python=3.11
conda activat xmo
pip install -r requirements.txt
```

## Create Dataset

```sh
python create_dataset.py --data_folder ./data/spectrograms --plot_folder ./data/plots
```