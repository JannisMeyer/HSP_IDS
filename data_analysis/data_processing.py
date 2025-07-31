from typing import List
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.discriminant_analysis import StandardScaler
import os
import pyarrow.parquet as pq
from pyarrow import float32, float64, bool_
import pyarrow as pa
import duckdb as ddb
from . import initial_data as i

USEFUL_FEATURES_START = 'conn_duration'

# region data acquiring -----------------------------------------------------------------------------------------------------------------------

def get_pca_reduced(df, n_components):

    # scale data
    scaler = StandardScaler()
    df = scaler.fit_transform(df)

    # reduce features using PCA
    #pca = PCA(n_components=n_components)
    pca = PCA(n_components=0.9)
    return pd.DataFrame(pca.fit_transform(df))

def read_parquet(path, nr_of_rows = 0, selected_columns = []):
    schema = pq.read_schema(path)
    all_cols = schema.names
    all_types = [schema.field(i).type for i in range(len(schema))]

    if not selected_columns:
        selected_columns = [
            col for col, typ in zip(all_cols[7:], all_types[7:]) 
            if typ in (float32(), float64(), bool_())
        ]

    pf = pq.ParquetFile(path)
    df = pd.DataFrame()

    if(nr_of_rows == 0):
        df = pf.read(columns = selected_columns).to_pandas()
    else:
        df = next(pf.iter_batches(batch_size = nr_of_rows, columns=selected_columns))
        df = pa.Table.from_batches([df]).to_pandas()
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(float)

    return df

# region feature vectors -------------------------------------------------------------------------------------------------------------------------------

def get_fvs_from_parquet(parquet_paths : List = [],
                         attack_types : List = [],
                         NR_ELEMENTS = 1,
                         all_samples : bool = False,
                         columns : List = []):
    fvs = pd.DataFrame()
    labels = pd.DataFrame()

    # create columns string for SQL SELECT
    schema = pq.read_schema('/home/hsp252/Development/DDoS/intrusion_normal_DDoS_1_final.parquet')
    all_cols = schema.names
    all_types = [schema.field(i).type for i in range(len(schema))]

    if not columns:
        columns = [
            col for col, type in zip(all_cols[7:], all_types[7:]) 
            if type in (float32(), float64(), bool_())
        ]
    columns = [f'"{col}"' for col in columns]
    columns = ", ".join(columns)

    # iterate over directories and query with SQL
    for i, attack_type in enumerate(parquet_paths): 
        if all_samples:
            fvs_local = ddb.query(f"""SELECT {columns} FROM '{attack_type}/*.parquet'""").to_df()
            fvs = pd.concat([fvs, fvs_local])
            labels = pd.concat([labels, pd.DataFrame({'attack_type': [attack_types[i]] * fvs_local.shape[0]})])
        else:
            fvs_local = ddb.query(f"""
                        SELECT {columns} FROM '{attack_type}/*.parquet'
                        USING SAMPLE {NR_ELEMENTS} ROWS
                    """).to_df()
            fvs = pd.concat([fvs, fvs_local])
            labels = pd.concat([labels, pd.DataFrame({'attack_type': [attack_types[i]] * fvs_local.shape[0]})])
            #print(f"{attack_types[i]}: {fvs_local.shape}")
    
    # convert bool to float
    bool_cols = fvs.select_dtypes(include='bool').columns
    fvs[bool_cols] = fvs[bool_cols].astype(float)

    return fvs, labels

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


# region data plotting -----------------------------------------------------------------------------------------------------------------------

def plot_correlation(plot_dendrogram : bool):

    # build df with samples from every class and compute correlation matrix
    df, unused = get_fvs_from_parquet(i.parquet_paths,
                                i.attack_types,
                                i.NR_MIN_ELEMENTS_IN_ALL_FILES,
                                False)
    # pd.set_option('display.max_rows', None)
    # print(df.isna().mean().sort_values(ascending=False) * 100)
    # print(f"Column with highest percentage of NAs: f{df.isna().sum().idxmax()},\
    #       {(df.isna().sum().max()) / (df.shape[0])}")
    # print(f"Percentage of NAs in df: {(df.isna().sum().sum()) / (df.shape[0] * df.shape[1])}")
    corr_matrix = df.corr().fillna(0)
    np.fill_diagonal(corr_matrix.values, 1)
    #print(f"Percentage of Nans in corr matrix: {(corr_matrix.isna().sum().sum()) / (corr_matrix.shape[0] * corr_matrix.shape[1])}")

    # create plot
    if plot_dendrogram:

        # compute hierarchical clusters
        distance_matrix = 1.0 - corr_matrix
        distance_matrix = np.clip(distance_matrix, a_min=0, a_max=100) # clip negative values
        condensed_distance_vector = squareform(distance_matrix.values)
        Z = linkage(condensed_distance_vector, method='average') # performs hierarchical clustering

        # plot dendrogram
        plt.figure(figsize=(20, 10))
        dendrogram(Z, labels=corr_matrix.columns.tolist(), leaf_rotation=90)
        plt.title("Feature Clustering Dendrogram")
        plt.show()

    else:

        # plot heat map
        plt.figure(figsize=(40, 40))
        sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.show()
    
def plot_feature_importances(important_features : list,
                             important_features_values : list,
                             s1_features : list = [],
                             s2_features : list = [],
                             s3_features : list = [],
                             connection_features : list = []):
    s1_mask = [i for i, feature in enumerate(important_features) if feature in s1_features]
    s2_mask = [i for i, feature in enumerate(important_features) if feature in s2_features]
    s3_mask = [i for i, feature in enumerate(important_features) if feature in s3_features]
    connection_mask = [i for i, feature in enumerate(important_features) if feature in connection_features]

    plt.figure(figsize=(12, 6))

    if s1_mask:
            plt.barh(important_features[s1_mask], important_features_values[s1_mask], color = 'red', label='s1')
    if s2_mask:
            plt.barh(important_features[s2_mask], important_features_values[s2_mask], color = 'blue', label='s2')
    if s3_mask:
            plt.barh(important_features[s3_mask], important_features_values[s3_mask], color = 'green', label='s3')
    if connection_mask:
            plt.barh(important_features[connection_mask], important_features_values[connection_mask], color = 'orange', label='connection')
    plt.xlabel("Importance")
    plt.title("Top 20 Most Important Features")
    plt.tight_layout()
    plt.legend()
    plt.show()

# region auxiliary ----------------------------------------------------------------------------------------------------------------------------

def get_parquet_row_nr(path : Path):
    if os.path.isdir(path):
        nr_rows = 0

        for file in os.scandir(path):
            file_path = Path(path) / file
            nr_rows = nr_rows + pq.ParquetFile(file_path).metadata.num_rows
        return nr_rows
    elif os.path.isfile(path):
        return pq.ParquetFile(path).metadata.num_rows
    else:
        raise Exception(f"\'{path}\' is not a directory nor a file or could not be found!")

