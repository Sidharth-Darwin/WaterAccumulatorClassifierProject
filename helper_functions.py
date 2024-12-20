import warnings
import pathlib
import pickle
import base64
import tempfile
import streamlit as st
from streamlit import cache_resource
from sklearn.ensemble import RandomForestClassifier
import umap
import matplotlib.pyplot as plt
from create_train_test_df import get_feature_label_df
warnings.filterwarnings("ignore")

label_to_character_mapping = {
    0: "०",
    1: "१",
    2: "२",
    3: "३",
    4: "४",
    5: "५",
    6: "६",
    7: "७",
    8: "८",
    9: "९",
}

colors = [
    "#E6194B",  # Red
    "#3CB44B",  # Green
    "#FFE119",  # Yellow
    "#4363D8",  # Blue
    "#F58231",  # Orange
    "#911EB4",  # Purple
    "#46F0F0",  # Cyan
    "#F032E6",  # Magenta
    "#BCF60C",  # Lime Green
    "#FFCC99"   # Peach
]

@cache_resource
def get_trained_model():
    if pathlib.Path("best_model").exists():
        with open("best_model", "rb") as f:
            return pickle.load(f)
    df = get_feature_label_df(train=True)
    X = df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]].to_numpy()
    y = df["labels"].to_numpy()
    clf = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=10, min_samples_leaf=8, max_depth=10, criterion="entropy")
    clf.fit(X, y)
    with open("best_model", "wb") as f:
        pickle.dump(clf, f)
    return clf

def predict_new_feature_vector(new_feature_vector):
    clf = get_trained_model()
    return clf.predict(new_feature_vector.reshape(1, -1)).item()

def visualize_feature_extraction(*args):
    n_rows = 2
    n_cols = 5
    rows = [st.container() for _ in range(n_rows)]
    cols_per_row = [r.columns(n_cols) for r in rows]
    cols = [column for row in cols_per_row for column in row]
    for i, simulation in enumerate(args):
        anim = simulation.visualize_traversal()
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            anim.save(f.name, writer="pillow", dpi=30, fps=50, savefig_kwargs={"facecolor": "white"})
            f.seek(0)
            with open(f.name, "rb") as temp_file:
                data = temp_file.read()
        data_url = base64.b64encode(data).decode()
        cols[i].markdown(f'<img src="data:image/gif;base64,{data_url}" alt="Animation GIF">', unsafe_allow_html=True)

def visualize_data_distribution(labels_to_include, new_feature_vector=None, n_samples=500):
    df = get_feature_label_df(train=False).sample(n=n_samples, random_state=42)
    X = df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]].to_numpy()
    y = df["labels"].to_numpy()
    umap_obj = get_umap_obj()
    X_umap = umap_obj.transform(X)
    plt.style.use("dark_background")
    fig, axs = plt.subplots(figsize=(10, 10))
    axs.axis("off")
    scatter_objs = []
    for i in labels_to_include:
        scatter_objs.append(axs.scatter(X_umap[y == i, 0], X_umap[y == i, 1], c=colors[i], s=25, label=str(i)))
    if new_feature_vector is not None:
        new_feature_vector = new_feature_vector.reshape(1, -1)
        new_feature_vector_tsne = umap_obj.transform(new_feature_vector)
        scatter_new_vec = axs.scatter(new_feature_vector_tsne[0, 0], new_feature_vector_tsne[0, 1], c="white", marker="X", edgecolors="white",s=200, label="Input data embedding")
    else:
        scatter_new_vec = None
    axs.legend()
    return fig, scatter_objs, scatter_new_vec

@cache_resource
def get_umap_obj():
    if pathlib.Path("saved_umap_obj").exists():
        with open("saved_umap_obj", "rb") as f:
            return pickle.load(f)
    df = get_feature_label_df(train=True)
    X = df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]].to_numpy()
    y = df["labels"].to_numpy()
    umap_obj = umap.UMAP(n_components=2, n_neighbors=15, metric="euclidean", output_metric="euclidean", random_state=42, n_jobs=1)
    umap_obj.fit(X, y)
    with open("saved_umap_obj", "wb") as f:
        pickle.dump(umap_obj, f)
    return umap_obj
