from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from . import data_processing as dp
# for later: k-fold cross-validation: split data into k subsets, train on k-1, test on 1
# -> do this k times and evaluate model
# Grid Search for hyperparameter tuning

def create_feature_vectors(thirtySecondWindow : dp.ThirtySecWindow, method : str = 'pca'):
     #TODO: 0/0 error for some connections when pca, get minimum number of ten-second windows
     main_df = dp.pd.DataFrame()

     # use PCA or autoencoder
     if method == 'pca':
        for host in thirtySecondWindow.hosts:
            most_common_tensecwindow_count_ = dp.most_common_tensecwindow_count(host)

            for connection in host.connections:
                len_ = len(connection.ten_sec_windows)
                #if len_ >= most_common_tensecwindow_count_ and len_ > 1:
                if len_ > 1:
                    df = dp.pd.DataFrame()
                    
                    # acquire data from all ten-second windows and averaged data
                    for ten_second_window in connection.ten_sec_windows:
                        df = dp.pd.concat([df, ten_second_window.data])
                    
                    averaged_tensecwindow_df_ = dp.averaged_tensecwindow_df(connection)
                    
                    # keep useful features only
                    idx = df.columns.get_loc(dp.USEFUL_FEATURES_START)
                    df = df.iloc[:, idx:]
                
                    # fill NaNs with averaged values
                    df = df.fillna(averaged_tensecwindow_df_.iloc[0])

                    # convert bools to floats
                    bool_cols = df.select_dtypes(include='bool').columns
                    df[bool_cols] = df[bool_cols].astype(float)

                    # reduce features
                    reduced_df = dp.get_reduced_features(df, most_common_tensecwindow_count_)
                    main_df = dp.pd.concat([main_df, reduced_df], ignore_index=True)
        return main_df
     elif method == 'autoencoder':
         # apply autoencoder to the data
         pass
     else:
         raise ValueError("Method must be either 'pca' or 'autoencoder'")
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