import cv2 as cv
import pandas as pd
from sklearn.utils import shuffle


def to_dict(id, z):
    return {
        "mask": cv.imread(f"train/images/{id}.png", cv.IMREAD_GRAYSCALE),
        "z": z,
        "id": id
    }


data = [to_dict(row[1], row[2]) for row in pd.read_csv("depths.csv").itertuples()]

# remove data entries with no image/mask
data = [img for img in data if img["mask"] is None]

shuffle(data)

df = pd.DataFrame(data)
df = df.drop("mask", axis=1)

df.to_csv('depth_test.csv', sep='\t', encoding='utf-8', index=False)

