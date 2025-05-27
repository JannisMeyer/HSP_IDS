from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from . import data_processing as dp
# for later: k-fold cross-validation: split data into k subsets, train on k-1, test on 1
# -> do this k times and evaluate model

def create_feature_vectors(thirtySecondWindow : dp.ThirtySecWindow):
     pass


class RFClassifier:
    def __init__(self, X, y):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10)
        self.X_train, self.X_test, self.y_train, self.y_test, self.predictions = self.train_model(X, y)

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
        self.model.fit(self.X_train, self.y_train)
        return X_train, X_test, y_train, y_test

    def predict(self, X):
        self.predictions = self.model.predict(X)
        print(self.predictions)