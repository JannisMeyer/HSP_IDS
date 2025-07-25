from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from . import data_processing as dp
from . import initial_data as i
# for later: k-fold cross-validation: split data into k subsets, train on k-1, test on 1
# -> do this k times and evaluate model
# Grid Search for hyperparameter tuning or Optuna

# region feature vectors -------------------------------------------------------------------------------------------------------------------------------

def create_hostbased_fvs_parquet(path : dp.Path, selected_columns : list): # pass absolute path!
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
            host_mean_connections = host.groupby(['conn_protocol', 'conn_src_ip', 'conn_dst_ip']) # this is key of a connection across 10s windows TODO: add ports
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

def create_hostbased_fvs_csv(tsw : dp.ThirtySecondWindow, selected_columns : list):
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

def create_and_store_host_based_fvs(data_set_path : dp.Path, ddos_test_path_parquet : dp.Path):
    feature_vectors = dp.pd.DataFrame()
    tsw_paths = dp.getThirtySecondWindowPaths(data_set_path)
    NR_OF_FVS = 100

    # create ddos feature vectors
    selected_cols = get_feature_names_from_csv('/home/hsp252/nas_mount/hunter.ids.data/hunter.ids.preprocessor/processed_dataframes/angriff/2025-02-28_08-41-32_192.168.1.0-normal_backdoor/1556466432.434372-1556466462.434372/192.168.1.152/connections/1556466432.434372-1556466442.434372/96.0_192.168.1.193_49338.0_192.168.1.152_1880.0/host_data_chunk_full.csv').select_dtypes(include='number').to_list() \
    + get_feature_names_from_csv('/home/hsp252/nas_mount/hunter.ids.data/hunter.ids.preprocessor/processed_dataframes/angriff/2025-02-28_08-41-32_192.168.1.0-normal_backdoor/1556466432.434372-1556466462.434372/192.168.1.152/s2_selected_qs.csv').select_dtypes(include='number').to_list() \
    + get_feature_names_from_csv('/home/hsp252/nas_mount/hunter.ids.data/hunter.ids.preprocessor/processed_dataframes/angriff/2025-02-28_08-41-32_192.168.1.0-normal_backdoor/1556466432.434372-1556466462.434372/192.168.1.152/s3_connection_qs.csv').select_dtypes(include='number').to_list() \
    + get_feature_names_from_csv('/home/hsp252/nas_mount/hunter.ids.data/hunter.ids.preprocessor/processed_dataframes/angriff/2025-02-28_08-41-32_192.168.1.0-normal_backdoor/1556466432.434372-1556466462.434372/s1_general_qs.csv').select_dtypes(include='number').to_list()

    test_ddos_feature_vectors = create_hostbased_fvs_parquet(ddos_test_path_parquet, selected_cols)

    test_ddos_feature_vectors_path = dp.Path(r'/home/hsp252/Development/HSP_IDS/test_ddos_df.pkl')
    save_to_pickle(test_ddos_feature_vectors, test_ddos_feature_vectors_path)

    ddos_feature_vectors = dp.pp.load(dp.Path('/home/hsp252/Development/HSP_IDS/test_ddos_df.pkl'))
    valid_columns = ddos_feature_vectors.columns.to_list()
    ddos_feature_vectors['attack_type'] = 'ddos'
    feature_vectors = dp.pd.concat([feature_vectors, ddos_feature_vectors])

    # create mitm feature vectors and store
    mitm_feature_vectors = dp.pd.DataFrame()
    for index, tsw_object in tsw_paths[tsw_paths['type'] == 'mitm'].iterrows():
        start = dp.time.time()
        tsw = dp.ThirtySecondWindow(dp.Path(tsw_object['path']))
        mitm_local_feature_vectors = create_hostbased_fvs_csv(tsw, valid_columns)
        mitm_local_feature_vectors['attack_type'] = tsw_object['type'] # add attack type column for training
        mitm_feature_vectors = dp.pd.concat([mitm_feature_vectors, mitm_local_feature_vectors])
        end = dp.time.time()
        row_count = mitm_feature_vectors.shape[0]
        print(f"created mitm feature vectors for {tsw_object['type']} in {end - start}s, rows: {row_count}")
        if row_count >= NR_OF_FVS:
            break
        #break
    mitm_feature_vectors_path = dp.Path(r'/home/hsp252/Development/HSP_IDS/test_mitm_df.pkl')
    save_to_pickle(mitm_feature_vectors, mitm_feature_vectors_path)

    # create runsomware feature vectors and store
    runsomware_feature_vectors = dp.pd.DataFrame()
    for index, tsw_object in tsw_paths[tsw_paths['type'] == 'runsomware'].iterrows():
        start = dp.time.time()
        tsw = dp.ThirtySecondWindow(dp.Path(tsw_object['path']))
        runsomware_local_feature_vectors = create_hostbased_fvs_csv(tsw, valid_columns)
        runsomware_local_feature_vectors['attack_type'] = tsw_object['type'] # add attack type column for training
        runsomware_feature_vectors = dp.pd.concat([runsomware_feature_vectors, runsomware_local_feature_vectors])
        end = dp.time.time()
        row_count = runsomware_feature_vectors.shape[0]
        print(f"created runsomware feature vectors for {tsw_object['type']} in {end - start}s, rows: {row_count}")
        if row_count >= NR_OF_FVS:
            break
        #break
    runsomware_feature_vectors_path = dp.Path(r'/home/hsp252/Development/HSP_IDS/test_runsomware_df.pkl')
    save_to_pickle(runsomware_feature_vectors, runsomware_feature_vectors_path)

    # create injection feature vectors and store
    injection_feature_vectors = dp.pd.DataFrame()
    for index, tsw_object in tsw_paths[tsw_paths['type'] == 'injection'].iterrows():
        start = dp.time.time()
        tsw = dp.ThirtySecondWindow(dp.Path(tsw_object['path']))
        injection_local_feature_vectors = create_hostbased_fvs_csv(tsw, valid_columns)
        injection_local_feature_vectors['attack_type'] = tsw_object['type'] # add attack type column for training
        injection_feature_vectors = dp.pd.concat([injection_feature_vectors, injection_local_feature_vectors])
        end = dp.time.time()
        row_count = injection_feature_vectors.shape[0]
        print(f"created injection feature vectors for {tsw_object['type']} in {end - start}s, rows: {row_count}")
        if row_count >= NR_OF_FVS:
            break
        #break
    injection_feature_vectors_path = dp.Path(r'/home/hsp252/Development/HSP_IDS/test_injection_df.pkl')
    save_to_pickle(injection_feature_vectors, injection_feature_vectors_path)

    # create backdoor feature vectors and store
    backdoor_feature_vectors = dp.pd.DataFrame()
    for index, tsw_object in tsw_paths[tsw_paths['type'] == 'backdoor'].iterrows():
        start = dp.time.time()
        tsw = dp.ThirtySecondWindow(dp.Path(tsw_object['path']))
        backdoor_local_feature_vectors = create_hostbased_fvs_csv(tsw, valid_columns)
        backdoor_local_feature_vectors['attack_type'] = tsw_object['type'] # add attack type column for training
        backdoor_feature_vectors = dp.pd.concat([backdoor_feature_vectors, backdoor_local_feature_vectors])
        end = dp.time.time()
        row_count = backdoor_feature_vectors.shape[0]
        print(f"created backdoorn feature vectors for {tsw_object['type']} in {end - start}s, rows: {row_count}")
        if row_count >= NR_OF_FVS:
            break
        #break
    backdoor_feature_vectors_path = dp.Path(r'/home/hsp252/Development/HSP_IDS/test_backdoor_df.pkl')
    save_to_pickle(backdoor_feature_vectors, backdoor_feature_vectors_path)

    # create dos feature vectors and store
    dos_feature_vectors = dp.pd.DataFrame()
    for index, tsw_object in tsw_paths[tsw_paths['type'] == 'dos'].iterrows():
        start = dp.time.time()
        tsw = dp.ThirtySecondWindow(dp.Path(tsw_object['path']))
        dos_local_feature_vectors = create_hostbased_fvs_csv(tsw, valid_columns)
        dos_local_feature_vectors['attack_type'] = tsw_object['type'] # add attack type column for training
        dos_feature_vectors = dp.pd.concat([dos_feature_vectors, dos_local_feature_vectors])
        end = dp.time.time()
        row_count = dos_feature_vectors.shape[0]
        print(f"created dos feature vectors for {tsw_object['type']} in {end - start}s, rows: {row_count}")
        if row_count >= NR_OF_FVS:
            break
        #break
    dos_feature_vectors_path = dp.Path(r'/home/hsp252/Development/HSP_IDS/test_dos_df.pkl')
    save_to_pickle(dos_feature_vectors, dos_feature_vectors_path)

    # create normal feature vectors and store
    normal_fvs = dp.pd.DataFrame()
    for index, tsw_object in tsw_paths[tsw_paths['type'] == 'normal'].iterrows():
        start = dp.time.time()
        tsw = dp.ThirtySecondWindow(dp.Path(tsw_object['path']))
        normal_local_fvs = create_hostbased_fvs_csv(tsw, valid_columns)
        normal_local_fvs['attack_type'] = tsw_object['type'] # add attack type column for training
        normal_fvs = dp.pd.concat([normal_fvs, normal_local_fvs])
        end = dp.time.time()
        row_count = normal_fvs.shape[0]
        print(f"created normal fv for {tsw_object['type']} in {end - start}s, rows: {row_count}")
        if row_count >= NR_OF_FVS:
            break
        #break
    normal_fv_path = dp.Path(r'/home/hsp252/Development/HSP_IDS/test_normal_df.pkl')
    save_to_pickle(normal_fvs, normal_fv_path)
    

# region classifiers -------------------------------------------------------------------------------------------------------------------------------

# TODO: look at statistics after training, evtl. LSTM oder Transformer
def rfc(fvs, labels):
        x_train, x_test, y_train, y_test = train_test_split(fvs, labels, stratify=labels)
        rfc = RandomForestClassifier(verbose=False, n_jobs=28)

        grid = {'n_estimators':[100, 500, 1000],
                #'max_depth':[20],
                'max_depth':[3, 5, 7, 10, 20, 25],
                #'min_samples_leaf':[1]}
                'min_samples_leaf':[1, 2]}
        gs = GridSearchCV(estimator=rfc,
                          param_grid=grid,
                          scoring='accuracy',
                          cv=3,
                          return_train_score=True,
                          verbose=False)
        gs.fit(x_train, y_train.values.ravel())

        best_rfc = gs.best_estimator_
        predictions = best_rfc.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        feature_importances = best_rfc.feature_importances_

        return best_rfc, gs.best_params_, predictions, accuracy, feature_importances

def i_forest(fvs, labels):

    # TODO: transform labels to -1,1 here before usage
    # use GridSearch for Hyperparameter Tuning (n_estimators, max_samples, max_features)
    # for contamination: set to auto, train, test on various data sets and look at decision function values
    # use multiple processors
    # save IsolationForest as pkl-file

    # def f1_scorer(y, y_pred):

    #     # transform to array of either -1 (attack) or 1 (normal) to fit to Isolation Forest prediciton output
    #     y = [-1 if x != 'normal' else 1 for x in y]
    #     y_pred = [-1 if x != 'normal' else 1 for x in y_pred]

    #     print(len(y))
    #     print(len(y_pred))

    #     return f1_score(y, y_pred)
    def f1_scorer(estimator, X, y_true, **kwargs):
        y_true = [-1 if x != 'normal' else 1 for x in y_true]
        y_pred = estimator.predict(X)
        # convert to 0/1 labels
        y_pred = (y_pred == -1).astype(int)
        y_true = (y_true == -1).astype(int)
        return f1_score(y_true, y_pred)

    i_forest = IsolationForest(verbose=False, n_jobs=28)

    grid = {'n_estimators':[50, 100, 500],
            #'max_samples':[256],
            'max_samples':[50, 256, 1000],
            #'max_features':[0.1]}
            'max_features':[0.05, 0.1, 0.5]}
    gs = GridSearchCV(estimator=i_forest,
                      param_grid=grid,
                      scoring=make_scorer(f1_scorer),
                      cv=3,
                      return_train_score=True, verbose=False)
    gs.fit(fvs, labels)

    best_i_forest = gs.best_estimator_

    return best_i_forest, gs.best_params_

def train_test_rfc(features, nr_training_samples, nr_test_samples):

        # train on passed features
        train_fvs, train_labels = dp.get_fvs_from_parquet(parquet_paths=i.parquet_paths,
                                                        NR_ELEMENTS=nr_training_samples,
                                                        attack_types=i.attack_types,
                                                        all_samples=False)

        train_fvs = train_fvs[features]

        best_rfc, best_params_before, predictions, train_accuracy, feature_importances = rfc(train_fvs, train_labels)

        # test on passed features
        test_fvs, test_labels = dp.get_fvs_from_parquet(parquet_paths=i.parquet_paths,
                                                                NR_ELEMENTS=nr_test_samples,
                                                                attack_types=i.attack_types,
                                                                all_samples=False)
        test_fvs = test_fvs[features]

        test_predictions = best_rfc.predict(test_fvs)
        test_accuracy = accuracy_score(test_labels, test_predictions)

        return train_accuracy, test_accuracy
    
# region auxiliary -------------------------------------------------------------------------------------------------------------------------------

def save_to_pickle(data, path : dp.Path):
    dp.pp.save(path, data, overwrite=True)
    print("saved data")

def get_feature_names_from_csv(path : dp.Path):
    return dp.pd.read_csv(path).columns