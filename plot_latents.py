from pathlib import Path
import numpy as np
import numpy.typing as npt
import umap
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from bokeh.plotting import figure, show, output_file
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
)
import bokeh.models as bmo
from bokeh.transform import factor_cmap
from bokeh.layouts import row, column
from bokeh.palettes import Category10, Category20


def embed_vectors(latents: npt.NDArray, seed: int = 42, model="umap") -> npt.NDArray:
    if model == "umap":
        reducer = umap.UMAP(random_state=seed)
        reducer.fit(latents)
        embedding = reducer.transform(latents)
    else:
        reducer = PCA(n_components=50)
        embedding = reducer.fit_transform(latents)
    return embedding


def cluster_latents(latents: npt.NDArray) -> npt.NDArray:
    gmm = DBSCAN()
    labels = gmm.fit_predict(latents)
    return labels


def plot_embedding(embedding: npt.NDArray, labels: npt.NDArray, paths: list[str]):
    output_file("data/latent_vectors/scatter_with_image_panel.html")

    labels = labels % 20

    # Data for the scatter plot and images
    data = {
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "images": paths,
        "labels": labels,
    }

    # Convert integer labels to strings for categorical mapping
    data["labels"] = list(map(str, data["labels"]))

    source = ColumnDataSource(data=data)

    labels = list(set(data["labels"]))
    colors = Category20[len(labels)]  # Use as many colors as there are labels

    # Scatter plot
    scatter_plot = figure(
        title="Scatter Plot",
        x_axis_label="X",
        y_axis_label="Y",
        tools="pan,wheel_zoom,box_zoom,reset,tap",
    )
    scatter_plot.circle(
        "x",
        "y",
        size=5,
        source=source,
        color=factor_cmap("labels", palette=colors, factors=labels),
    )

    # Div to display the image
    image_div = Div(width=1000, height=1000)

    # JavaScript callback to update the image
    callback = CustomJS(
        args=dict(source=source, image_div=image_div),
        code="""
            const indices = cb_data.index.indices;
            if (indices.length > 0) {
                const index = indices[0];
                const img_url = source.data.images[index];
                image_div.text = '<img src="' + img_url + '" style="width:100%;height:100%;">';
            } else {
                image_div.text = '';
            }
        """,
    )

    # HoverTool to trigger the callback on hover
    hover_tool = HoverTool(tooltips=None, callback=callback, mode="mouse")
    scatter_plot.add_tools(hover_tool)

    # Layout the scatter plot and image panel
    layout = row(scatter_plot, image_div)
    show(layout)


def main():
    output_folder = Path("data/latent_vectors")
    path = output_folder / "latents.npy"
    latents = np.load(path)

    path = output_folder / "shot_ids.npy"
    shot_ids = np.load(path)
    img_paths = [
        f"../../data/plots/{shot}.omaha_3lz.png" for shot in shot_ids.squeeze()
    ]

    print(latents.shape)
    embedding = embed_vectors(latents, model="pca")
    print(embedding.shape)
    embedding = embed_vectors(embedding, model="umap")
    labels = cluster_latents(embedding)
    plot_embedding(embedding, labels, img_paths)


if __name__ == "__main__":
    main()
