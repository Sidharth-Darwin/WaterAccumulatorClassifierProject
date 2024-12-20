from PIL import Image
import pathlib
from streamlit import cache_data
import numpy as np
import pandas as pd
import tqdm
from movement_simulators import ExecuteAllPourMovements
from feature_extractors import FeatureExtractor
from download_data import download_data

@cache_data
def get_feature_label_df(train=True):
    if train and pathlib.Path("train_features_labels.csv").exists():
        return pd.read_csv("train_features_labels.csv")
    if not train and pathlib.Path("test_features_labels.csv").exists():
        return pd.read_csv("test_features_labels.csv")
    all_paths = download_data()
    if train:
        path = all_paths["TRAIN_DATA_PATH"]
    else:
        if pathlib.Path("test_features_labels.csv").exists():
            return pd.read_csv("test_features_labels.csv")
        path = all_paths["TEST_DATA_PATH"]
    path = pathlib.Path(path).absolute()
    feature_extractor = FeatureExtractor()
    pour_movements = ExecuteAllPourMovements(display_traversal=False)
    df_dict = {"labels": [], "image_path": []}
    for label in tqdm.tqdm(path.iterdir()):
        if not label.is_dir():
            continue
        for image_file in label.iterdir():
            if not image_file.is_file() and image_file.as_posix().endswith(".png"):
                continue
            df_dict["image_path"].append(image_file.as_posix())
            df_dict["labels"].append(int(label.name))
    df = pd.DataFrame(df_dict)
    df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]] = np.nan
    for row in tqdm.tqdm(df.itertuples(), total=len(df)):
        image_file = getattr(row, 'image_path')
        image = Image.open(image_file).convert("1")
        image = np.asarray(image, dtype=bool)
        features = feature_extractor.extract_features(image, *pour_movements.get_objects(image))
        df.loc[row.Index, ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]] = features
    del df["image_path"]
    if train:
        df.to_csv("train_features_labels.csv", index=False)
    else:
        df.to_csv("test_features_labels.csv", index=False)
    return df
