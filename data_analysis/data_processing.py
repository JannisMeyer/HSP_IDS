import time
from typing import List
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.discriminant_analysis import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from collections import defaultdict
import pypickle as pp
import os
import pyarrow.parquet as pq
from pyarrow import float32, float64, bool_
import pyarrow as pa
import gc
import dask.dataframe as dd
import duckdb as ddb

# TODO: Feature-Reduktion (Clustering) für einzelne Verbindungen
# TODO: rebuild data structure and functions

# region classes -------------------------------------------------------------------------------------------------------------------------------

#  TODO: have a df of all host_data_chunk_full.csvs combined and use that, maybe only useable if > 1
class Connection:
    def __init__(self, path : Path):
        self.data = self.get_data(path / 'host_data_chunk_full.csv')

    def get_data(self, data_path):
        df = pd.read_csv(data_path)

        # convert bools to floats
        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(float)

        return df   

# TODO: consider removing this, as it is somewhat constant, and just have connections per host
# class TenSecondWindow:
#     def __init__(self, path : Path):
#         self.connections = self.get_connections(path)

#     def get_connections(self, path):
#         connections : list[Connection] = [] # imply data type so list is iterable and doesn't recall the get-method

#         for entry in os.listdir(path):
#             entry_path = path / entry
#             if entry_path.is_dir():
#                 connections.append(Connection(entry_path))

#         return connections

# TODO: consider having averaged stuff here to access it directly
class Host:
    def __init__(self, path : Path, names : List):
        self.s2 = self.get_s2(path / 's2_selected_qs.csv')
        self.s3 = self.get_s3(path / 's3_connection_qs.csv')
        #self.ten_second_windows = self.get_ten_second_windows(path)
        self.connections : defaultdict = self.get_connections(path, names)

    def get_s2(self, path : Path):
        return pd.read_csv(path)

    def get_s3(self, path : Path):
        return pd.read_csv(path)

    # def get_ten_second_windows(self, path):
    #     ten_second_windows : list[TenSecondWindow] = []
    #     ten_second_windows_path = path / 'connections'

    #     if ten_second_windows_path.exists():
    #         for entry in os.listdir(ten_second_windows_path):
    #             entry_path = ten_second_windows_path / entry

    #             if entry_path.is_dir():
    #                 ten_second_windows.append(TenSecondWindow(entry_path))

    #     return ten_second_windows
    
    def get_connections(self, path: Path, names : List):
        path = path / 'connections'
        connections = defaultdict(pd.DataFrame)

        # go over all 10s windows and collect respective connection data
        for ten_second_window in os.scandir(path):
            ten_second_window_path = path / ten_second_window

            #if ten_second_window_path.is_dir(): # is_dir() is slow
            for i, connection in enumerate(os.scandir(ten_second_window_path)):
                connection_path = ten_second_window_path / connection

                #if connection_path.is_dir():
                connections[connection.name] = pd.concat([connections[connection.name], pd.read_csv(connection_path / 'host_data_chunk_full.csv')]) # use connection.name, otherwise it doesn't count as same key!
        return connections


class ThirtySecondWindow:
    def __init__(self, path : Path):
        self.s1 = self.get_s1(path.joinpath('s1_general_qs.csv'))
        self.connection_feature_names = pd.read_csv('connection_features.CSV').columns
        self.hosts = self.get_hosts(path)

    def get_s1(self, s1_path):
        return pd.read_csv(s1_path)

    def get_hosts(self, path : Path):
        hosts : list[Host] = []

        for entry in os.scandir(path):
            entry_path = path / entry

            if entry_path.is_dir():
                hosts.append(Host(entry_path, self.connection_feature_names))

        return hosts

# region data plotting -----------------------------------------------------------------------------------------------------------------------
USEFUL_FEATURES_START = 'conn_duration'

# TODO: adapt to actual data structure (collect time series of connections over 10s windows and do correlation)
def plot_correlation(thirty_sec_window : ThirtySecondWindow, plot_dendrogram : bool):

    for host_index, host in enumerate(thirty_sec_window.hosts):
        print(f"\n----- Host {host_index + 1} -----")

        all_cors = []
        
        for ten_second_window in host.ten_second_windows:
            feature_series = {}

            if ten_second_window.connections.__len__() > 1:
                for connection in ten_second_window.connections:
                    
                    # on first run, determine which columns to use
                    if not feature_series:
                        all_columns = list(connection.data.columns)
                        start_index = all_columns.index(USEFUL_FEATURES_START)
                        useful_features = all_columns[start_index:]
                        feature_series = {feature: [] for feature in useful_features}
                    
                    # get values of features
                    for feature in feature_series:
                        value = connection.data[feature][0]
                        value = float(value) # cast to float for consistency (bools)
                        feature_series[feature].append(value)

                # build DataFrame from collected features
                df = pd.DataFrame(feature_series)
                corr = df.corr(method='pearson')
                all_cors.append(corr)
            else:
                print("Not enough 10s-Windows to compute correlation, skipping!")

        # stack into 3D array
        if all_cors.__len__() > 0:
            stacked = np.stack([corr.values for corr in all_cors])

            # compute the mean and variance over the z-axis
            mean_corr_matrix = np.mean(stacked, axis=0)
            #var_matrix = np.var(stacked, axis=0)

            # retrieve labels
            features = all_cors[0].columns

            # wrap back into DataFrames
            mean_corr_df = pd.DataFrame(mean_corr_matrix, index=features, columns=features)
            #var_df = pd.DataFrame(var_matrix, index=features, columns=features)

            # create plot
            mean_corr_df_na_dropped = mean_corr_df.corr().dropna(how="all").dropna(axis=1, how="all")

            if plot_dendrogram:

                # compute hierarchical clusters
                distance_matrix = 1 - mean_corr_df_na_dropped
                distance_matrix = np.clip(distance_matrix, a_min=0, a_max=100) # clip negative values
                condensed_distance_vector = squareform(distance_matrix.values)
                Z = linkage(condensed_distance_vector, method='average') # performs hierarchical clustering

                # plot dendrogram
                plt.figure(figsize=(10, 5))
                dendrogram(Z, labels=mean_corr_df_na_dropped.columns.tolist(), leaf_rotation=90)
                plt.title("Feature Clustering Dendrogram")
                plt.show()

            else:

                # plot heat map
                plt.figure(figsize=(20, 20))
                sns.heatmap(mean_corr_df_na_dropped, annot=False, fmt=".2f", cmap="coolwarm", center=0)
                plt.title("Feature Correlation Heatmap")
                plt.tight_layout()
                plt.show()
        else:
            print(f"No correlation data for host {host_index + 1}!")

        #break

# TODO: adapt to actual data structure (collect time series of connections over 10s windows and do kmeans)
def connection_kmeans(thirty_second_window : ThirtySecondWindow, n_clusters : int = 3):

    for host in thirty_second_window.hosts:

        print(f"\n----- Host {host.s2['sq_identifier'][0]} -----")

        for id, ten_second_window in enumerate(host.ten_second_windows):
            print(f"----- 10s-Window {id} -----")
            features = []
            columns = None

            if len(ten_second_window.connections) > 1:
                average_connection = get_average_connection(ten_second_window)

                # cut useless features
                columns = ten_second_window.connections[0].data.columns
                start_idx = columns.get_loc(USEFUL_FEATURES_START)
                
                for connection in ten_second_window.connections:
                    row = connection.data

                    # fill NaNs with averaged values
                    row = row.fillna(average_connection.iloc[0])

                    # acquire usefull features
                    row = row.values[0]
                    useful_values = row[start_idx:]
                    useful_values = np.array(useful_values, dtype=float).reshape(-1, 1)
                    features.append(useful_values)

                # convert to np.array and remove NaNs
                features = np.array(features, dtype=float)

                # rotate axes to have shape (n_features, n_tensecwindows=12, 1)
                features = np.swapaxes(features, 0, 1) 

                # -> have np.array with shape (n_features, n_tensecwindows=12, 1)
                # -> every feature will be assigned to a cluster

                # fit samples
                features_scaled = TimeSeriesScalerMeanVariance().fit_transform(features)

                # compute cluster
                model = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=0)
                labels = model.fit_predict(features_scaled)

                print("Inertia: "+str(model.inertia_))

                # create df for plotting
                df = pd.DataFrame({
                    'Feature': columns[start_idx:],
                    'Cluster': labels
                })

                counts = df.value_counts('Cluster').reset_index(name='Count')

                # plot bar chart for clusters
                plt.figure(figsize=(6, 6))
                plt.bar(counts['Cluster'].astype(str), counts['Count'], color='skyblue')
                plt.ylabel('Count')
                plt.title('Feature Clusters from KMeans')
                plt.tight_layout()
                plt.show()
            else:
                print("Not enough connections to compute clustering, skipping!")
                continue
            break
        break

def time_series_kmeans(thirty_second_window : ThirtySecondWindow, n_clusters : int = 3):

    for host in thirty_second_window.hosts:

        print(f"\n----- Host {host.s2['sq_identifier'][0]} -----")
        samples = []

        # get most common 10s window count for maximum data availability
        most_common_tensecwindow_count_ = get_most_common_connection_count(host)
        print(f"Most common 10s window count is {most_common_tensecwindow_count_}, using that for clustering")
        
        if most_common_tensecwindow_count_ > 1: # less than 2 is not applicable
            for ten_second_window in host.ten_second_windows:
                if len(ten_second_window.connections) == most_common_tensecwindow_count_:
                    sample = []
                    average_connection = get_average_connection(ten_second_window)

                    # cut useless features
                    columns = ten_second_window.connections[0].data.columns
                    start_idx = columns.get_loc(USEFUL_FEATURES_START)
                    
                    for connection in ten_second_window.connections:
                        row = connection.data

                        # fill NaNs with averaged values
                        row = row.fillna(average_connection.iloc[0])

                        # acquire usefull features
                        row = row.values[0]
                        useful_values = row[start_idx:]
                        sample.append(useful_values)

                    # convert to np.array and remove NaNs
                    sample = np.array(sample, dtype=float)
                    sample = sample[~np.isnan(sample)]
                    samples.append(sample)
            
            # -> have np.array with shape (n_connections, n_tensecwindows, n_features)
            # -> every connection will be assigned to a cluster

            # disregard empty samples
            if len(samples) > 0:

                # fit samples
                X = np.array(samples)
                X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)

                # compute cluster
                model = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=0)
                labels = model.fit_predict(X_scaled)

                print("Inertia: "+str(model.inertia_))
                print(labels)
            else:
                print("No samples to cluster for this host!")
            #break
        else:
            print("Not enough connections to compute clustering, skipping!")
            #break


# region data acquiring -----------------------------------------------------------------------------------------------------------------------

def get_s1_features(path : Path): # use path for multi 30s window analysis
    return pd.read_csv(path / 's1_general_qs.csv')

def get_s2_features(path : Path): # use path for multi 30s window analysis
    df_list = []

    # iterate through hosts and acquire respective s2s
    for entry in os.listdir(path):
        df = pd.DataFrame()
        entry_path = path / entry

        if entry_path.is_dir():
            row = pd.read_csv(entry_path / 's2_selected_qs.csv')
            df = pd.concat([df, row], ignore_index=True)
        
        df_list.append(df)

    return df_list

def get_s3_features(path : Path): # use path for multi 30s window analysis
    df_list = []

    # iterate through hosts and acquire respective s3s
    for entry in os.listdir(path):
        df = pd.DataFrame()
        entry_path = path / entry

        if entry_path.is_dir():
            row = pd.read_csv(entry_path / 's3_connection_qs.csv')
            df = pd.concat([df, row], ignore_index=True)

        df_list.append(df)

    return df_list

# TODO: adapt to actual data structure (collect time series of connections over 10s windows and reduce)
def get_connection_features(thirty_second_window : ThirtySecondWindow, get_reduced : bool):
    host_dict = {}

    # iterate over hosts
    for host in thirty_second_window.hosts:
        connection_dict = {}

        # iterate over connections
        for i, ten_second_window in enumerate(host.ten_second_windows):
            print(f"---- Processing connection {i} ----")

            # iterate over ten second windows and acquire connection features
            if len(ten_second_window.connections) > 1:
                df = pd.DataFrame()
                average_connection = get_average_connection(ten_second_window)
                
                for connection in ten_second_window.connections:
                    df = pd.concat([df, connection.data])
                
                # keep only columns starting from 'conn_duration'
                df_all = df_all.loc[:, USEFUL_FEATURES_START:]
            
                # get features directly or in reduced form
                if get_reduced:

                    # fill NaNs with averaged values
                    df = df.fillna(average_connection.iloc[0])

                    # convert bools to floats
                    bool_cols = df.select_dtypes(include='bool').columns
                    df[bool_cols] = df[bool_cols].astype(float)
                    #print(df)

                    # reduce features using PCA
                    connection_dict[ten_second_window] = get_reduced(df)
                    print(connection_dict[ten_second_window])
                else:
                    connection_dict[ten_second_window] = df

                # -> have dict of hosts with dict of 10s windows with df of connections with respective features
                host_dict[host] = connection_dict
                #break
            else:
                #print("Not enough connections to compute reduced features, skipping!")
                pass
        break

    return host_dict

def get_pca_reduced(df, n_components):

    # scale data
    scaler = StandardScaler()
    df = scaler.fit_transform(df)

    # reduce features using PCA
    #pca = PCA(n_components=n_components)
    pca = PCA(n_components=0.9)
    return pd.DataFrame(pca.fit_transform(df))

def getThirtySecondWindowPaths(data_set_path : Path):
    thirty_second_windows = pd.DataFrame(columns=["path", "type"])

    # get attack tsw paths
    for entry in os.scandir(data_set_path / 'angriff'):
        entry_path = data_set_path / entry

        #if entry_path.is_dir():

        # get type of attack
        entry_lower = entry.name.lower()
        type = ''

        if "_dos" in entry_lower:
            type = "dos"
        elif "runsomware" in entry_lower:
            type = "runsomware"
        elif "backdoor" in entry_lower:
            type = "backdoor"
        elif "mitm" in entry_lower:
            type = "mitm"
        elif "_ddos" in entry_lower:
            type = "ddos"
        elif "injection" in entry_lower:
            type = "injection"
        else:
            type = "unknown"
        
        # get paths and add to df
        for window in os.scandir(entry_path):
            window_path = entry_path / window

            if window_path.is_dir():
                thirty_second_windows = pd.concat([
                    thirty_second_windows,
                    pd.DataFrame([{"path": str(window_path), "type": type}])
                ], ignore_index=True)
    
    # get normal tsw paths
    for entry in os.scandir(data_set_path / 'normal'):
        entry_path = data_set_path / entry

        # get paths and add to df
        if entry_path.is_dir():
            for window in os.scandir(entry_path):
                window_path = entry_path / window

                if window_path.is_dir():
                    thirty_second_windows = pd.concat([
                        thirty_second_windows,
                        pd.DataFrame([{"path": str(window_path), "type": 'normal'}])
                    ], ignore_index=True)
    thirty_second_windows.reset_index()

    return thirty_second_windows

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

def get_fvs_from_parquet(parquet_paths : List,
                         NR_ELEMENTS,
                         attack_types : List,
                         all_samples : bool,
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
            #print(f"{attack_types[i]}: {fvs_local.shape}")
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

# region auxiliary ----------------------------------------------------------------------------------------------------------------------------

# dummy
def feed_to_model(df : pd.DataFrame):
    print(df.iloc[0][0])

def load_windows_one_by_one(data_path : Path):

    # yield one feature df by one, instead of returning one huge aggregated df
    for window in os.listdir(data_path):
        window_path = data_path / window
        yield get_connection_features(window_path)

def get_most_common_connection_count(host : Host):
    counts = [len(ten_second_window.connections) for ten_second_window in host.ten_second_windows]

    if not counts:
        return None
    
    return max(set(counts), key=counts.count)

# def get_average_connection(ten_second_window: TenSecondWindow):

#     # collect all dfs
#     dfs = [connection.data for connection in ten_second_window.connections]

#     df_all = pd.concat(dfs, ignore_index=True)

#     # remove useless features
#     df_all = df_all.loc[:, USEFUL_FEATURES_START:]
    
#     # compute mean
#     avg = df_all.mean(axis=0, skipna=True)
    
#     # replace NaN means with 0 if all values are NaN
#     avg_filled = avg.fillna(0)
    
#     return pd.DataFrame([avg_filled])

def average_features(thirtySecondWindow : ThirtySecondWindow):

    # collect all ten-second windows of 30s window
    global_df = pd.DataFrame()
    
    for host in thirtySecondWindow.hosts:
        for ten_second_window in host.ten_second_windows:
            for connection in ten_second_window.connections:
                global_df = pd.concat([global_df, connection.data], ignore_index=True)
    
    # keep useful features only
    global_df = global_df.loc[:, USEFUL_FEATURES_START:]

    #compute mean
    global_df = global_df.mean(axis=0, skipna=True)

    # replace NaN means with 0 if all values are NaN
    global_df = global_df.fillna(0)

    return global_df # return series so that df.fillna(global_df) works correctly

