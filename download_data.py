import kagglehub
import pathlib

def download_data():
    path = kagglehub.dataset_download("imbikramsaha/hindi-mnist")
    path = pathlib.Path(path).absolute()
    all_paths = {"BASE_PATH": path, "TRAIN_DATA_PATH": path / "Hindi-MNIST" / "train", "TEST_DATA_PATH": path / "Hindi-MNIST" / "test"}
    return all_paths