import numpy as np
import xarray as xr
import intake
import pandas as pd
from pathlib import Path
from datasets import Dataset


def generate_dataset():
    catalog = intake.open_catalog("https://mastapp.site/intake/catalog.yml")
    bucket = catalog.level1.shots
    df = pd.DataFrame(catalog.index.level1.sources.read())

    xmo_df = df.loc[df.name == "xmo"]
    xmo_df = xmo_df.sort_values("shot_id", ascending=False)
    # xmo_df = xmo_df.iloc[:100]

    for _, row in xmo_df.iterrows():
        dataset = bucket(url=row.url).to_dask()
        if "omaha_3lz" in dataset:
            signal = dataset["omaha_3lz"].values
            shot = dataset.attrs["shot_id"]
            yield {"signal": signal, "shot_id": shot}


def create_dataset():
    ds = Dataset.from_generator(generate_dataset)
    return ds
