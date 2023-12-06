from xgboost import XGBClassifier
import numpy as np


class XGBoostClassifier:

    def __init__(self, np_array_path):
        data = np.load(np_array_path)
        self.X_train = data[:, :-1]
        self.y_train = data[:, -1]

    def train_model(self):
        xgbc = XGBClassifier(
            objective='multi:softprob',  # Use 'multi:softprob' for multiclass problems
            num_class=3,  # Number of classes in the classification problem
            eval_metric='mlogloss'  # Use 'mlogloss' for multiclass logloss
        )

        xgbc.fit(self.X_train, self.y_train)
        self.classifier = xgbc

    def evaluate_train_accuracy(self):
        if not hasattr(self, 'classifier'):
            print('Cannot evaluate accuracy before training the model')
            return 0

        predictions_matrix = self.classifier.predict(self.X_train)
        predictions_vector = [[index for index, value in enumerate(list_m) if value == 1] for list_m in predictions_matrix]
        predictions_vector = [el for list_pred in predictions_vector for el in list_pred]

        return np.mean(self.y_train == predictions_vector)

    def predict_class(self, X_sample):
        if X_sample.ndim == 1:
            X_sample = [X_sample]
        return self.classifier.predict_proba(X_sample)


if __name__ == "__main__":
    classifier = XGBoostClassifier("../data/training_data.npy")

    classifier.train_model()

    print("The XGBoost accuracy over the train sample is:", classifier.evaluate_train_accuracy())
