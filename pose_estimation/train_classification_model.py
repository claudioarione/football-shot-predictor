import json
from xgboost import XGBClassifier
import numpy as np

if __name__ == "__main__":
    data = np.load("../data/training_data.npy")
    print(data)
    X_train = data[:, :-1]
    y_train = data[:, -1]
    print(X_train)
    print(y_train)
    xgbc = XGBClassifier(n_estimators=100, max_depth=1, learning_rate=0.15)
    xgbc.fit(X_train, y_train)

    acc_test_xgbc = np.mean(y_train == xgbc.predict(X_train))

    print("The XGBoost accuracy over the test sample is:", acc_test_xgbc)
    print(X_train[:1, :])
    print(xgbc.predict(X_train[1:2, :]))
