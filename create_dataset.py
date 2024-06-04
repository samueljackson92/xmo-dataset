import numpy as np
import intake
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.signal import stft
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging

logging.basicConfig(level=logging.INFO)


def create_spectrogram(dataset: xr.Dataset):
    ds = dataset["omaha_3lz"]
    # Parameters to limit the number of frequencies
    nperseg = 2000  # Number of points per segment
    nfft = 2000  # Number of FFT points

    # Compute the Short-Time Fourier Transform (STFT)
    sample_rate = 1 / (ds.time[1] - ds.time[0])
    f, t, Zxx = stft(ds, fs=int(sample_rate), nperseg=nperseg, nfft=nfft)

    dataset = xr.Dataset(
        dict(spectrogram=(("frequency", "time"), Zxx), frequency=f, time=t)
    )
    return dataset


def plot_spectrogram(spectogram: xr.Dataset, output_file: str):
    fig, ax = plt.subplots(figsize=(15, 5))
    cax = ax.pcolormesh(
        spectogram.time,
        spectogram.frequency / 1000,
        np.abs(spectogram.spectrogram),
        shading="nearest",
        cmap="jet",
        norm=LogNorm(vmin=1e-5),
    )
    ax.set_ylim(0, 50)
    ax.set_title(f'XMO/OMAHA/3LZ - Shot {spectogram.attrs["shot_id"]}')
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [sec]")
    plt.colorbar(cax, ax=ax)
    plt.savefig(output_file)


def main():
    data_folder = Path("data/test")
    data_folder.mkdir(parents=True, exist_ok=True)

    plot_folder = Path("data/plots")
    plot_folder.mkdir(parents=True, exist_ok=True)

    catalog = intake.open_catalog("catalog.yml")
    bucket = catalog["stfc_s3"]

    df = pd.read_parquet(Path("~/Downloads/sources.parquet").expanduser())
    xmo_df = df.loc[df.source == "xmo"]

    for _, row in xmo_df.iterrows():
        output_file = data_folder / f"{row.shot_id}.zarr"
        if output_file.exists():
            continue

        logging.info(f"Getting XMO data for {row.url}")
        dataset = bucket(url=row.url).to_dask()

        spectrogram = create_spectrogram(dataset)
        spectrogram.attrs["shot_id"] = row.shot_id

        spectrogram.to_zarr(output_file, mode="w", consolidated=True)

        output_file = plot_folder / f"{row.shot_id}.png"
        plot_spectrogram(spectrogram, output_file)


if __name__ == "__main__":
    main()
