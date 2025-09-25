from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

def train_model(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train class distribution:", pd.Series(y_train).value_counts().to_dict())
    print("Test class distribution :", pd.Series(y_test).value_counts().to_dict())

    lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)

    print("Number of features model expects:", lr.coef_.shape[1])
    print("Model classes:", lr.classes_)

    return lr, X_train, X_test, y_train, y_test
