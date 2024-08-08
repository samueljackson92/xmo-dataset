import matplotlib.pyplot as plt
from src.train_classifier import SpectrogramDataset, load_annotations


def test_dataloader():
    df = load_annotations("data/annotations.csv")
    dataset = SpectrogramDataset(df.file_name.values, df.annotations.values)
    spec, label = dataset[0]
    print(spec.min(), spec.max())
    plt.matshow(spec)
    plt.title(f"{label}")
    plt.savefig("test.png")
    assert label
