from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from . import data_processing as dp
# for later: k-fold cross-validation: split data into k subsets, train on k-1, test on 1
# -> do this k times and evaluate model
# Grid Search for hyperparameter tuning

def create_feature_vectors(thirtySecondWindow : dp.ThirtySecWindow, method : str = 'pca'):
     main_df = dp.pd.DataFrame()

     # use PCA or autoencoder
     if method == 'pca': #TODO: include s1, s2, s3

        # collect all 10s windows of 30s window
        global_df = dp.pd.DataFrame()
        for host in thirtySecondWindow.hosts:
            for connection in host.connections:
                for ten_second_window in connection.ten_sec_windows:
                    global_df = dp.pd.concat([global_df, ten_second_window.data], ignore_index=True)
        
        # keep useful features only
        global_df = global_df.loc[:, dp.USEFUL_FEATURES_START:]

        # fill NaNs with averaged values
        global_averaged_df = dp.average_features(thirtySecondWindow)
        global_df = global_df.fillna(global_averaged_df)

        # convert bools to float
        bool_cols = global_df.select_dtypes(include='bool').columns
        global_df[bool_cols] = global_df[bool_cols].astype(float)

        # scale
        global_scaler = dp.StandardScaler()
        global_df_scaled = global_scaler.fit_transform(global_df)

        # fit pca
        global_pca = dp.PCA(n_components=20)
        global_pca.fit(global_df_scaled)

        # apply global scaling and pca to all ten-second windows and collect
        main_pca_df = dp.pd.DataFrame()
        for host in thirtySecondWindow.hosts:
            for connection in host.connections:
                features_average = dp.averaged_tensecwindow_df(connection)

                for ten_second_window in connection.ten_sec_windows:

                    # keep useful features only
                    features = ten_second_window.data.loc[:, dp.USEFUL_FEATURES_START:]

                    # fill NaNs with averaged values
                    features = features.fillna(features_average)

                    # convert bools to float
                    bool_cols = features.select_dtypes(include='bool').columns
                    features[bool_cols] = features[bool_cols].astype(float)

                    # scale
                    features_scaled = global_scaler.transform(features)

                    # apply pca
                    features_pca = global_pca.transform(features_scaled)

                    # concatenate
                    main_pca_df = dp.pd.concat([main_pca_df, dp.pd.DataFrame(features_pca)], ignore_index=True)

        # return mean df
        return dp.pd.DataFrame(main_pca_df.mean(axis=0))

     elif method == 'autoencoder': # TODO: autoencoder
         # apply autoencoder to the data
         return main_df
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