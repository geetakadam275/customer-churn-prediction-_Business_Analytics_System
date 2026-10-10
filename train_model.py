import json
from config import DATA_PATH, METRICS_PATH
from src.data_utils import load_data
from src.ml_pipeline import train_models

if __name__ == "__main__":
    results, best, _, _ = train_models(load_data(DATA_PATH))
    print("Best model by F1-score:", best)
    for name, m in results.items():
        print(name, m)
    print("Detailed metrics saved to", METRICS_PATH)
