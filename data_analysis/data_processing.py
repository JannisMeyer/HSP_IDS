import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# (TODO: create feature vectors, later)
# TODO: make yielding faster
# TODO: Feature-Reduktion (Clustering)
# TODO: PCA: create data structure to contain reductions for all hosts and connections of a 30s window
# TODO: Notizen für Projekt-Bericht anfangen

# region classes -------------------------------------------------------------------------------------------------------------------------------

class TenSecWindow:
    def __init__(self, path : Path):
        self.data = self.getData(path / 'host_data_chunk_full.csv')

    def getData(self, data_path):
        return pd.read_csv(data_path)

class Connection:
    def __init__(self, path : Path):
        self.tenSecWindows = self.getTenSecWindows(path)

    def getTenSecWindows(self, path):
        tenSecWindows : list[TenSecWindow] = [] # imply data type so list is iterable and doesn't recall the get-method

        for entry in os.listdir(path):
            entry_path = path / entry
            if entry_path.is_dir():
                tenSecWindows.append(TenSecWindow(entry_path))

        return tenSecWindows

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

def plot_correlation(thirty_sec_window : ThirtySecWindow, plot_dendrogram : bool):
    
    USEFUL_FEATURES_START = 'conn_duration'

    for host_index, host in enumerate(thirty_sec_window.hosts):
        print(f"\n----- Host {host_index + 1} -----")

        all_cors = []
        
        for connection in host.connections:
            feature_series = {}

            if connection.tenSecWindows.__len__() > 1:
                for ten_sec_window in connection.tenSecWindows:
                    
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

def get_connection_features(thirtySecondWindow : ThirtySecWindow, get_reduced : bool):
    host_dict = {}

    # iterate over hosts
    for host in thirtySecondWindow.hosts:
        connection_dict = {}

        # iterate over connections
        for connection in host.connections:

            # iterate over ten second windows and acquire connection features
            if len(connection.tenSecWindows) > 1:
                df = pd.DataFrame()
                
                for tenSecondWindow in connection.tenSecWindows:
                    df = pd.concat([df, tenSecondWindow.data])
                
                # keep only columns starting from 'conn_duration'
                idx = df.columns.get_loc('conn_duration')
                df = df.iloc[:, idx:]
            
                # get features directly or in reduced form
                if get_reduced:
                    df.dropna(axis=1, inplace=True)
                    df = df.loc[:, df.dtypes != bool]
                    connection_dict[connection] = get_reduced_feature_df(df)
                else:
                    connection_dict[connection] = df

                # -> have dict of hosts with dict of connections with df of 10s-windows with respective features
                host_dict[host] = connection_dict
            else:
                print("Not enough 10s windows to compute reduced features, skipping!")

    return host_dict

def get_reduced_features(df):
    # TODO: clustering, mit mehr oder weniger Dimensionen, dann plotten, evtl. skalieren
    # TODO: 30s-Fenster vergleichen hinsichtlich s1, s2, s3, aber erst von unten, Connections, Hosts, 30s-Fenster
    # Fragen: Sieht man Unterschiede im Cluster zwischen verschiedenen Angriffen?
    #         Werden die Cluster besser, wenn man erst PCA macht?
    #         Kann man die Korrelationen irgendwie aufbrechen?
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

