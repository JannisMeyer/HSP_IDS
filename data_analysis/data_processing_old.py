from typing import List
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from collections import defaultdict
import os


# region classes -------------------------------------------------------------------------------------------------------------------------------

class Connection:
    def __init__(self, path : Path):
        self.data = self.get_data(path / 'host_data_chunk_full.csv')

    def get_data(self, data_path):
        df = pd.read_csv(data_path)

        # convert bools to floats
        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(float)

        return df   

class TenSecondWindow:
    def __init__(self, path : Path):
        self.connections = self.get_connections(path)

    def get_connections(self, path):
        connections : list[Connection] = [] # imply data type so list is iterable and doesn't recall the get-method

        for entry in os.listdir(path):
            entry_path = path / entry
            if entry_path.is_dir():
                connections.append(Connection(entry_path))

        return connections

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

    def get_ten_second_windows(self, path):
        ten_second_windows : list[TenSecondWindow] = []
        ten_second_windows_path = path / 'connections'

        if ten_second_windows_path.exists():
            for entry in os.listdir(ten_second_windows_path):
                entry_path = ten_second_windows_path / entry

                if entry_path.is_dir():
                    ten_second_windows.append(TenSecondWindow(entry_path))

        return ten_second_windows
    
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


# region data plotting -----------------------------------------------------------------------------------------------------------------------

USEFUL_FEATURES_START = 'conn_duration'

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


# region auxiliary ----------------------------------------------------------------------------------------------------------------------------

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

def get_average_connection(ten_second_window: TenSecondWindow):

    # collect all dfs
    dfs = [connection.data for connection in ten_second_window.connections]

    df_all = pd.concat(dfs, ignore_index=True)

    # remove useless features
    df_all = df_all.loc[:, USEFUL_FEATURES_START:]
    
    # compute mean
    avg = df_all.mean(axis=0, skipna=True)
    
    # replace NaN means with 0 if all values are NaN
    avg_filled = avg.fillna(0)
    
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