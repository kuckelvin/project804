import pandas as pd
import numpy as np

dataset = pd.read_csv("../data/raw/dataset.csv")

dataset.describe()
dataset.describe(include=object)

dataset["Target"]

dataset["Target"].value_counts()

dataset.isna().sum()

features = dataset.columns

X = dataset.drop("Target", axis=1)
y = dataset["Target"]
