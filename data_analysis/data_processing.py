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

# TODO: Feature-Reduktion (Clustering) für einzelne Verbindungen

# region classes -------------------------------------------------------------------------------------------------------------------------------

class TenSecWindow:
    def __init__(self, path : Path):
        self.data = self.get_data(path / 'host_data_chunk_full.csv')

    def get_data(self, data_path):
        df = pd.read_csv(data_path)

        # convert bools to floats
        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(float)

        return df   

class Connection:
    def __init__(self, path : Path):
        self.ten_sec_windows = self.getTenSecWindows(path)

    def getTenSecWindows(self, path):
        ten_sec_windows : list[TenSecWindow] = [] # imply data type so list is iterable and doesn't recall the get-method

        for entry in os.listdir(path):
            entry_path = path / entry
            if entry_path.is_dir():
                ten_sec_windows.append(TenSecWindow(entry_path))

        return ten_sec_windows

class Host:
    def __init__(self, path : Path):
        self.s2 = self.get_s2(path / 's2_selected_qs.csv')
        self.s3 = self.get_s3(path / 's3_connection_qs.csv')
        self.connections = self.get_connections(path)

    def get_s2(self, s2_path):
        return pd.read_csv(s2_path)

    def get_s3(self, s3_path):#
        return pd.read_csv(s3_path)

    def get_connections(self, path):
        connections : list[Connection] = []
        connections_path = path / 'connections'

        if connections_path.exists():
            for entry in os.listdir(connections_path):
                entry_path = connections_path / entry

                if entry_path.is_dir():
                    connections.append(Connection(entry_path))

        return connections

class ThirtySecWindow:
    def __init__(self, path : Path):
        self.s1 = self.get_s1(path / 's1_general_qs.csv')
        self.hosts = self.get_hosts(path)

    def get_s1(self, s1_path):
        return pd.read_csv(s1_path)

    def get_hosts(self, path):
        hosts : list[Host] = []

        for entry in os.listdir(path):
            entry_path = path / entry

            if entry_path.is_dir():
                hosts.append(Host(entry_path))

        return hosts
    

# region data plotting -----------------------------------------------------------------------------------------------------------------------
USEFUL_FEATURES_START = 'conn_duration'

# TODO: test this on real datasets
def plot_correlation(thirty_sec_window : ThirtySecWindow, plot_dendrogram : bool):

    for host_index, host in enumerate(thirty_sec_window.hosts):
        print(f"\n----- Host {host_index + 1} -----")

        all_cors = []
        
        for connection in host.connections:
            feature_series = {}

            if connection.ten_sec_windows.__len__() > 1:
                for ten_sec_window in connection.ten_sec_windows:
                    
                    # on first run, determine which columns to use
                    if not feature_series:
                        all_columns = list(ten_sec_window.data.columns)
                        start_index = all_columns.index(USEFUL_FEATURES_START)
                        useful_features = all_columns[start_index:]
                        feature_series = {feature: [] for feature in useful_features}
                    
                    # get values of features
                    for feature in feature_series:
                        value = ten_sec_window.data[feature][0]
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

def connection_kmeans(thirty_second_window : ThirtySecWindow, n_clusters : int = 3):

    for host in thirty_second_window.hosts:

        print(f"\n----- Host {host.s2['sq_identifier'][0]} -----")

        for id, connection in enumerate(host.connections):
            print(f"----- Connection {id} -----")
            features = []
            columns = None

            if len(connection.ten_sec_windows) > 1:
                averaged_tensecwindow_df_ = averaged_tensecwindow_df(connection)

                # cut useless features
                columns = connection.ten_sec_windows[0].data.columns
                start_idx = columns.get_loc(USEFUL_FEATURES_START)
                
                for ten_second_window in connection.ten_sec_windows:
                    row = ten_second_window.data

                    # fill NaNs with averaged values
                    row = row.fillna(averaged_tensecwindow_df_.iloc[0])

                    # acquire usefull features
                    row = row.values[0]
                    useful_values = row[start_idx:]
                    useful_values = np.array(useful_values, dtype=float).reshape(-1, 1)
                    features.append(useful_values)

                # convert to np.array and remove NaNs
                features = np.array(features, dtype=float)

                # rotate axes to have shape (n_features, n_tensecwindows, 1)
                features = np.swapaxes(features, 0, 1) 

                # -> have np.array with shape (n_features, n_tensecwindows, 1)
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
                print("Not enough 10s windows to compute clustering, skipping!")
                continue
            break
        break
    

# TODO: test this on real datasets
# TODO: redo for a single connection with normal Kmeans, compare before and after PCA
# requirements txt (Anforderungsliste) erstellen für das Training und an Murad schicken
# Welche Modelle bieten sich an für das Training?
# Modellstruktur pitchen
# host-weise
def time_series_kmeans(thirty_second_window : ThirtySecWindow, n_clusters : int = 3):

    for host in thirty_second_window.hosts:

        print(f"\n----- Host {host.s2['sq_identifier'][0]} -----")
        samples = []

        # get most common 10s window count for maximum data availability
        most_common_tensecwindow_count_ = most_common_tensecwindow_count(host)
        print(f"Most common 10s window count is {most_common_tensecwindow_count_}, using that for clustering")
        
        if most_common_tensecwindow_count_ > 1: # less than 2 is not applicable
            for connection in host.connections:
                if len(connection.ten_sec_windows) == most_common_tensecwindow_count_:
                    sample = []
                    averaged_tensecwindow_df_ = averaged_tensecwindow_df(connection)

                    # cut useless features
                    columns = connection.ten_sec_windows[0].data.columns
                    start_idx = columns.get_loc(USEFUL_FEATURES_START)
                    
                    for ten_second_window in connection.ten_sec_windows:
                        row = ten_second_window.data

                        # fill NaNs with averaged values
                        row = row.fillna(averaged_tensecwindow_df_.iloc[0])

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
            print("Not enough 10s windows to compute clustering, skipping!")
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

def get_connection_features(thirty_second_window : ThirtySecWindow, get_reduced : bool):
    host_dict = {}

    # iterate over hosts
    for host in thirty_second_window.hosts:
        connection_dict = {}

        # iterate over connections
        for i, connection in enumerate(host.connections):
            print(f"---- Processing connection {i} ----")

            # iterate over ten second windows and acquire connection features
            if len(connection.ten_sec_windows) > 1:
                df = pd.DataFrame()
                averaged_tensecwindow_df_ = averaged_tensecwindow_df(connection)
                
                for ten_second_window in connection.ten_sec_windows:
                    df = pd.concat([df, ten_second_window.data])
                
                # keep only columns starting from 'conn_duration'
                idx = df.columns.get_loc(USEFUL_FEATURES_START)
                df = df.iloc[:, idx:]
            
                # get features directly or in reduced form
                if get_reduced:

                    # fill NaNs with averaged values
                    df = df.fillna(averaged_tensecwindow_df_.iloc[0])

                    # convert bools to floats
                    bool_cols = df.select_dtypes(include='bool').columns
                    df[bool_cols] = df[bool_cols].astype(float)
                    #print(df)

                    # reduce features using PCA
                    connection_dict[connection] = get_reduced_features(df)
                    print(connection_dict[connection])
                else:
                    connection_dict[connection] = df

                # -> have dict of hosts with dict of connections with df of 10s-windows with respective features
                host_dict[host] = connection_dict
                break
            else:
                #print("Not enough 10s windows to compute reduced features, skipping!")
                pass
        break

    return host_dict

def get_reduced_features(df):
    # TODO: clustering, mit mehr oder weniger Dimensionen, dann plotten, evtl. skalieren
    # TODO: 30s-Fenster vergleichen hinsichtlich s1, s2, s3, aber erst von unten, Connections, Hosts, 30s-Fenster
    # Fragen: Sieht man Unterschiede im Cluster zwischen verschiedenen Angriffen?
    #         Werden die Cluster besser, wenn man erst PCA macht?
    #         Kann man die Korrelationen irgendwie aufbrechen?

    # scale data
    scaler = StandardScaler()
    df = scaler.fit_transform(df)

    # reduce features using PCA
    pca = PCA(n_components=0.95)
    return pd.DataFrame(pca.fit_transform(df))


# region auxiliary ----------------------------------------------------------------------------------------------------------------------------

# dummy
def feed_to_model(df : pd.DataFrame):
    print(df.iloc[0][0])

def load_windows_one_by_one(data_path : Path):

    # yield one feature df by one, instead of returning one huge aggregated df
    for window in os.listdir(data_path):
        window_path = data_path / window
        yield get_connection_features(window_path)

def most_common_tensecwindow_count(host : Host):
    counts = [len(conn.ten_sec_windows) for conn in host.connections]

    if not counts:
        return None
    
    return max(set(counts), key=counts.count)

def averaged_tensecwindow_df(connection: Connection):

    # collect all dfs
    dfs = [tsw.data for tsw in connection.ten_sec_windows]

    if not dfs:
        return pd.DataFrame([0], columns=[])

    df_all = pd.concat(dfs, ignore_index=True)

    # remove useless features
    columns = df_all.columns
    start_idx = columns.get_loc(USEFUL_FEATURES_START)
    df_all = df_all[start_idx:]
    
    # compute mean
    avg = df_all.mean(axis=0, skipna=True)
    
    # replace NaN means with 0 if all values are NaN
    avg_filled = avg.fillna(0)
    
    return pd.DataFrame([avg_filled])

