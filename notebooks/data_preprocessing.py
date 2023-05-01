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

# FEATURE SELECTION
y.shape

y.columns
type(y)
type(X)
y = pd.DataFrame(y)
y.value_counts()

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


#encoding the target variable
le = LabelEncoder()
y = le.fit_transform(y)


#doing the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#defining the pipeline steps
steps = [
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('skb', SelectKBest(score_func=chi2, k=10)),
    ('classifier', RandomForestClassifier(random_state=42))
]

#creating pipeline
pipeline = Pipeline(steps)

#defining hyperparameters for grid search
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

#performing grid search (with cross-validation)
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

#printing the best parameters and accuracy
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

#evaluating the model on the test set
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
