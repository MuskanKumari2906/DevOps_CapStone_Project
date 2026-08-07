import urllib.request
import os

url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
os.makedirs("data", exist_ok=True)
filepath = os.path.join("data", "heart.csv")

urllib.request.urlretrieve(url, filepath)
print(f"Downloaded dataset to {filepath}")
