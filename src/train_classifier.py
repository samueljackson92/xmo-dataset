import matplotlib.pyplot as plt
from torchmetrics import F1Score
from pathlib import Path
import numpy as np
import json
import argparse
import torch
import xarray as xr
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import ASTForAudioClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import lightning as L


class SpectrogramDataset(Dataset):
    def __init__(self, file_names, labels, max_size: int = 1024):
        self.file_names = file_names
        self.labels = labels
        self.max_size = max_size

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        dataset = xr.open_dataset(self.file_names[index])
        annotations = self.labels[index]
        spectrogram = dataset["spectrogram"].values

        ntime = len(spectrogram)

        label = np.zeros(len(spectrogram))

        for annotation in annotations:
            lindex = annotation.lower_index / 100
            lindex = int(ntime * lindex)
            uindex = annotation.upper_index / 100
            uindex = int(ntime * uindex)
            label[lindex:uindex] = 1

        if ntime < self.max_size:
            # Right pad the spectrogram
            spectrogram = np.pad(spectrogram, [(0, self.max_size - ntime), (0, 0)])
            label = np.pad(label, [(0, self.max_size - ntime)])
            # label = float(len(annotations) > 0)
        elif ntime >= self.max_size:
            # Randomly subsample spectrogram on time axis
            index = np.random.randint(0, ntime - self.max_size)
            spectrogram = spectrogram[index : index + self.max_size]
            label = label[index : index + self.max_size]
            # annotations_in_window = []
            # for annotation in annotations:
            #     if (
            #         annotation.upper_index >= index
            #         or annotation.lower_index <= index + self.max_size
            #     ):
            #         annotations_in_window.append(annotation)

            # label = float(len(annotations_in_window) > 0)

        assert (
            spectrogram.shape[0] == self.max_size
        ), f"Number of spectrogram time points must be {self.max_size}"

        return torch.tensor(spectrogram, dtype=torch.float), torch.tensor(
            label, dtype=torch.float
        )


class SpectogramClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_classes: int = 2,
        hidden_size: int = 100,
        train_transformer: bool = False,
    ) -> None:
        super().__init__()
        self.model = ASTForAudioClassification.from_pretrained(model_name)

        if not train_transformer:
            for param in self.model.audio_spectrogram_transformer.parameters():
                param.requires_grad = False

        last_layer_size = self.model.config.hidden_size
        self.model.num_labels = num_classes
        self.model.classifier = nn.Sequential(
            nn.Linear(last_layer_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_f1 = F1Score(task="multilabel", num_labels=num_classes)
        self.valid_f1 = F1Score(task="multilabel", num_labels=num_classes)

    def training_step(self, batch, batch_idx):
        spec, label = batch
        out = self.model(input_values=spec, labels=label)
        logits = out.logits
        out.loss = self.criterion(logits, label)
        loss = out.loss

        self.train_f1(logits, label)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        spec, label = batch
        out = self.model(input_values=spec, labels=label)
        logits = out.logits
        out.loss = self.criterion(logits, label)
        loss = out.loss

        self.valid_f1(logits, label)

        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        spec, label = batch
        out = self.model(input_values=spec, labels=label)
        logits = out.logits

        return {
            "score": nn.functional.sigmoid(logits),
            "spec": spec,
            "label": label,
        }

    def on_train_epoch_end(self):
        self.log("train_f1_epoch", self.train_f1, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("valid_f1_epoch", self.valid_f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class SpectrogramAnnotation:
    def __init__(self, x: int, width: int, label: str):
        self.lower_index = x
        self.upper_index = x + width
        self.label = label


def load_annotations(file_name: str):
    df = pd.read_csv(file_name)
    df["file_name"] = df.image.map(
        lambda x: x.replace("s3://xmo/", "data/spectrograms/").replace(".png", ".zarr")
    )
    df["label"] = df["label"].map(lambda x: x if isinstance(x, str) else "[]")
    df["label"] = df["label"].map(json.loads)
    df["has_modes"] = df["label"].map(lambda x: len(x) > 0)

    labels = df["label"].map(
        lambda annotations: [
            SpectrogramAnnotation(x["x"], x["width"], x["rectanglelabels"][0])
            for x in annotations
        ]
    )
    df["annotations"] = labels
    return df


def make_dataloader(file_names, labels, batch_size: int, shuffle: bool = False):
    dataset = SpectrogramDataset(file_names, labels)
    dataset = SpectrogramDataset(file_names, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def prediction_plots(trainer, model, dataloader, folder):
    results = trainer.predict(model, dataloader)
    log_dir = Path(trainer.log_dir) / folder
    log_dir.mkdir(exist_ok=True)
    for i, result in enumerate(results):
        for j, (spec, score, label) in enumerate(
            zip(result["spec"], result["score"], result["label"])
        ):
            fig, axes = plt.subplots(2, 1, figsize=(15, 5))
            ax = axes[0]
            ax.matshow(spec.T)

            ax = axes[1]
            ax.plot(label, marker=".")
            ax.plot(score, marker=".")
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlim(0, 1024)
            plt.tight_layout()
            plt.savefig(log_dir / f"plot_{i}_{j}.png")
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--train-transformer", action="store_true")
    args = parser.parse_args()

    seed = args.seed
    epochs = args.epochs
    batch_size = args.batch_size
    num_classes = 1024

    annotations_file = "data/annotations.csv"
    df = load_annotations(annotations_file)

    df_train, df_test = train_test_split(df, stratify=df.has_modes, random_state=seed)

    train_dataloader = make_dataloader(
        df_train.file_name.values, df_train.annotations.values, batch_size, shuffle=True
    )

    test_dataloader = make_dataloader(
        df_test.file_name.values, df_test.annotations.values, batch_size, shuffle=False
    )

    model = SpectogramClassifier(
        num_classes=num_classes, train_transformer=args.train_transformer
    )

    trainer = L.Trainer(max_epochs=epochs)
    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.validate(model, test_dataloader)
    prediction_plots(trainer, model, train_dataloader, "train")
    prediction_plots(trainer, model, test_dataloader, "test")


if __name__ == "__main__":
    main()
