import argparse
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


def create_spectrogram(dataset: xr.Dataset) -> xr.Dataset:
    # Parameters to limit the number of frequencies
    nperseg = 2000  # Number of points per segment
    nfft = 2000  # Number of FFT points

    # Compute the Short-Time Fourier Transform (STFT)
    sample_rate = 1 / (dataset.time[1] - dataset.time[0])
    f, t, Zxx = stft(dataset, fs=int(sample_rate), nperseg=nperseg, nfft=nfft)

    dataset = xr.Dataset(
        dict(spectrogram=(("frequency", "time"), Zxx), frequency=f, time=t)
    )
    return dataset


def plot_spectrogram(spectogram: xr.Dataset, output_file: str):
    _, ax = plt.subplots(figsize=(15, 5))
    cax = ax.pcolormesh(
        spectogram.time,
        spectogram.frequency / 1000,
        np.abs(spectogram),
        shading="nearest",
        cmap="jet",
        norm=LogNorm(vmin=1e-5),
    )
    ax.set_ylim(0, 50)
    ax.set_title(f'{spectogram.attrs["name"]} - Shot {spectogram.attrs["shot_id"]}')
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [sec]")
    plt.colorbar(cax, ax=ax)
    plt.savefig(output_file)


def create_spectrograms(
    dataset: xr.Dataset, metadata: dict, channels: list[str] = ["omaha_3lz"]
):
    spectrograms = []
    for channel, ds in dataset.data_vars.items():
        if channel.endswith("lz") and channel in channels:
            spectrogram = create_spectrogram(ds)
            spectrogram.spectrogram.attrs["shot_id"] = metadata.shot_id
            spectrogram.spectrogram.attrs["name"] = channel
            spectrogram = spectrogram.rename({"spectrogram": channel})
            spectrograms.append(spectrogram)

    spectrogram = xr.merge(spectrograms)
    return spectrogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="data/spectrograms")
    parser.add_argument("--plot_folder", default="data/plots")
    parser.add_argument("--plot", default=True)
    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)

    plot_folder = Path(args.plot_folder)
    plot_folder.mkdir(parents=True, exist_ok=True)

    catalog = intake.open_catalog("catalog.yml")
    bucket = catalog["stfc_s3"]

    df = pd.read_parquet(Path("~/Downloads/sources.parquet").expanduser())
    xmo_df = df.loc[df.source == "xmo"]
    xmo_df = xmo_df.sort_values("shot_id")

    for _, row in xmo_df.iterrows():
        logging.info(f"Getting XMO data for {row.url}")
        output_file = data_folder / f"{row.shot_id}.zarr"
        if output_file.exists():
            logging.info(f"Skipping {row.url} as {output_file} already exists")
            continue

        dataset = bucket(url=row.url).to_dask()
        spectrograms = create_spectrograms(dataset, row)
        spectrograms.to_zarr(output_file, mode="w", consolidated=True)

        if args.plot:
            for key in spectrograms.data_vars.keys():
                plot_file = plot_folder / f"{row.shot_id}.{key}.png"
                plot_spectrogram(spectrograms[key], plot_file)


if __name__ == "__main__":
    main()
