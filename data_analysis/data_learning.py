from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from . import data_processing as dp
# for later: k-fold cross-validation: split data into k subsets, train on k-1, test on 1
# -> do this k times and evaluate model
# Grid Search for hyperparameter tuning or Optuna

# region feature vectors -------------------------------------------------------------------------------------------------------------------------------

def create_test_ddos_feature_vectors(path : dp.Path, selected_columns : list): # pass absolute path!
    start = dp.time.time()
    test_ddos_parquet = dp.pd.read_parquet(path / 'intrusion_normal_DDoS_10_final.parquet')
    end = dp.time.time()
    print('time to read parquet ddos file: ' + str(end - start) + 's')

    feature_vectors = dp.pd.DataFrame()

    # group df by window time key and iterate so not to mix up 30s windows
    learning_dataset = dp.pd.DataFrame()
    test_ddos_parquet_window_time_key = test_ddos_parquet.groupby('window_time_key')

    for i, tsw in test_ddos_parquet_window_time_key:
        start = dp.time.time()

        # get connections
        test_ddos_parquet_conn_src = tsw.reset_index().groupby('conn_src_ip')
        test_ddos_parquet_conn_dst = tsw.reset_index().groupby('conn_dst_ip')

        for key in test_ddos_parquet_conn_src.groups.keys() & test_ddos_parquet_conn_dst.groups.keys():

            # get all connections for a host
            host = dp.pd.DataFrame()
            host = dp.pd.concat([host, test_ddos_parquet_conn_src.get_group(key)], axis=0, ignore_index=True)
            host = dp.pd.concat([host, test_ddos_parquet_conn_dst.get_group(key)], axis=0, ignore_index=True)
            host.reset_index()

            # get average connection values
            bool_columns = host.select_dtypes(include='bool').columns
            host[bool_columns] = host[bool_columns].astype(float)
            mean_host = host.select_dtypes(include='number').mean().fillna(0)

            # groupy by connection and mean
            host_mean_connections = host.groupby(['conn_protocol', 'conn_src_ip', 'conn_dst_ip']) # this is key of a connection across 10s windows
            feature_vector = dp.pd.DataFrame()
            
            for i, group in host_mean_connections:
                group = group.select_dtypes(include='number')
                group = group.fillna(mean_host).mean().to_frame().transpose()
                feature_vector = dp.pd.concat([feature_vector, group])
            
            # create feature vector
            feature_vector = feature_vector.median().to_frame().transpose().drop('index', axis=1)
            feature_vector = feature_vector[[col for col in selected_columns if col in feature_vector.columns]] # some cols in parquet files don't exist in CSVs
            feature_vectors = dp.pd.concat([feature_vectors, feature_vector])

            #break
        end = dp.time.time()
        print(f'created feature vector for another tsw, execution time: {end - start}s')
        #break
    return feature_vectors

def create_feature_vectors(tsw : dp.ThirtySecondWindow, selected_columns : list):
    feature_vectors = dp.pd.DataFrame()

    # go over connections of each host and get connections
    for host in tsw.hosts:

        # get average connection
        average_df = dp.pd.DataFrame()
        for connection in host.connections.values():
            all_data = dp.pd.concat([connection, tsw.s1, host.s2, host.s3], axis=1)
            average_df = dp.pd.concat([average_df, all_data])
        average_df = average_df[selected_columns]
        average_df = average_df.mean().to_frame().transpose() # deals with booleans
        average_df = average_df.fillna(0)

        # get connection features
        feature_vector = dp.pd.DataFrame()
        for connection in host.connections.values():

            # get all features
            connection_df = dp.pd.concat([connection, tsw.s1, host.s2, host.s3], axis=1)
            connection_df = (connection_df[selected_columns])

            # append meaned connection
            
            mean_connection = connection_df.fillna(average_df).mean().to_frame().transpose()
            feature_vector = dp.pd.concat([feature_vector, mean_connection])
            #break

        # create feature vector
        feature_vector = feature_vector.median().to_frame().transpose()
        feature_vectors = dp.pd.concat([feature_vectors, feature_vector])
        #break
    return feature_vectors

# region classifiers -------------------------------------------------------------------------------------------------------------------------------

# TODO: look at statistics after training, evtl. LSTM oder Transformer
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
    
# region auxiliary -------------------------------------------------------------------------------------------------------------------------------

def save_to_pickle(data, path : dp.Path):
    dp.pp.save(path, data, overwrite=True)
    print("saved data")