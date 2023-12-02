from xgboost import XGBClassifier
import numpy as np


class XGBoostClassifier:

    def __init__(self, np_array_path):
        data = np.load(np_array_path)
        self.X_train = data[:, :-1]
        self.y_train = data[:, -1]

    def train_model(self, n_estimators=100, max_depth=1, learning_rate=0.15):
        xgbc = XGBClassifier(n_estimators, max_depth, learning_rate)
        xgbc.fit(self.X_train, self.y_train)
        self.classifier = xgbc

    def evaluate_accuracy(self):
        if not hasattr(self, 'classifier'):
            print('Cannot evaluate accuracy before training the model')
            return 0
        return np.mean(self.y_train == self.classifier.predict(self.X_train))

    def predict_class(self, X_sample):
        return self.classifier.predict(X_sample)


if __name__ == "__main__":
    classifier = XGBoostClassifier("../data/training_data.npy")

    classifier.train_model(n_estimators=100, max_depth=1, learning_rate=0.15)

    print("The XGBoost accuracy over the test sample is:", classifier.evaluate_accuracy())
