import numpy as np
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForAudioClassification, ASTFeatureExtractor
from torch.utils.data import DataLoader
from data import create_dataset


class SpectogramEncoder:
    def __init__(self) -> None:
        # Load the pretrained AST model
        self.model = AutoModelForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        num_classes = 3
        self.model.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        self.sampling_rate = 100000
        self.processor = ASTFeatureExtractor(
            feature_size=1, sampling_rate=self.sampling_rate, max_length=1024
        )

    @torch.no_grad
    def encode(self, dataloader: DataLoader):
        latents = []
        shot_ids = []
        for item in dataloader:
            shot = item["shot_id"]
            print(shot)
            signal = item["signal"]
            spectrogram = self.processor(
                signal.squeeze(), return_tensors="pt", sampling_rate=self.sampling_rate
            )
            out = self.model(**spectrogram, output_hidden_states=True)
            latent = out.hidden_states[-1].reshape(1, -1)
            # latent = out.logits.numpy()
            latents.append(latent)
            shot_ids.append(shot)
        return np.concatenate(latents, axis=0), shot_ids


def main():
    output_folder = Path("data/latent_vectors")
    output_folder.mkdir(exist_ok=True, parents=True)

    dataset = create_dataset()
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=1)

    encoder = SpectogramEncoder()
    latents, shot_ids = encoder.encode(dataloader)

    file_name = output_folder / "latents.npy"
    np.save(file_name, latents)
    file_name = output_folder / "shot_ids.npy"
    np.save(file_name, shot_ids)


if __name__ == "__main__":
    main()
