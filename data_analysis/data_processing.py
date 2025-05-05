import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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

# region data analysis functions --------------------------------------------------------------------------------------------------------------

# (TODO: create feature vectors, later)
# TODO: make yielding faster
# TODO: Feature-Reduktion (PCA und Clustering)
# TODO: see if other hosts have similar heat maps
# TODO: setup Git

def show_s1_features(window : ThirtySecWindow):
    print(window.s1.columns)

def show_s2_features(window : ThirtySecWindow):
    print(window.hosts[0].s2.columns)

def show_s3_features(window : ThirtySecWindow):
    print(window.hosts[0].s3.columns)

def plot_s2_feature(window : ThirtySecWindow, feature : str):

    # collect data
    x_labels = []
    y_values = []

    for i, host in enumerate(window.hosts):
        try:
            value = host.s2[feature].iloc[0]  # get an element instead of the whole column
            x_labels.append(f'host{i+1}')
            y_values.append(value)
        except KeyError:
            print(f"'{feature}' not found in s2 of host{i+1}")
        except Exception as e:
            print(f"Error processing host{i+1}: {e}")

    # plot
    plt.figure(figsize=(10, 6))
    plt.bar(x_labels, y_values)
    plt.title(f'{feature} per Host (from s2)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_s3_feature(window : ThirtySecWindow, feature : str):

    # collect data
    x_labels = []
    y_values = []

    for i, host in enumerate(window.hosts):
        try:
            value = host.s3[feature].iloc[0]  # get an element instead of the whole column
            x_labels.append(f'host{i+1}')
            y_values.append(value)
        except KeyError:
            print(f"'{feature}' not found in s3 of host{i+1}")
        except Exception as e:
            print(f"Error processing host{i+1}: {e}")

    # plot
    plt.figure(figsize=(10, 6))
    plt.bar(x_labels, y_values)
    plt.title(f'{feature} per Host (from s3)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(thirty_sec_window : ThirtySecWindow):
    
    USEFUL_FEATURES_START = 'conn_duration'

    for host_index, host in enumerate(thirty_sec_window.hosts):
        print(f"\n----- Host {host_index + 1} -----")

        all_cors = []
        
        for conn_index, connection in enumerate(host.connections):
            #print(f"\n--- Connection {conn_index + 1} ---")
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
                #print(df.info())
                # if host_index is 0:
                #     print(feature_series["conn_duration"])

                # compute pairwise correlations
                corr = df.corr(method='pearson')
                #if(host_index == 3):
                    #print(corr.info())
                #print(corr)
                all_cors.append(corr)
            else:
                print("Not enough 10s-Windows to compute correlation, skipping!")

        # stack into 3D array
        if all_cors.__len__() > 0:
            stacked = np.stack([corr.values for corr in all_cors])

            # compute the mean and variance over the z-axis
            mean_corr_matrix = np.mean(stacked, axis=0)
            var_matrix = np.var(stacked, axis=0)

            # retrieve labels
            features = all_cors[0].columns

            # wrap back into DataFrames
            mean_corr_df = pd.DataFrame(mean_corr_matrix, index=features, columns=features)
            var_df = pd.DataFrame(var_matrix, index=features, columns=features)

            # create heatmap
            mean_corr_df_na_dropped = mean_corr_df.corr().dropna(how="all").dropna(axis=1, how="all")

            plt.figure(figsize=(20, 20))
            sns.heatmap(mean_corr_df_na_dropped, annot=False, fmt=".2f", cmap="coolwarm", center=0)
            plt.title("Feature Correlation Heatmap")
            plt.tight_layout()
            plt.show()
        else:
            print(f"No correlation data for host {host_index + 1}!")

        # filter and print variances and correlations
        #print(f"\n--- Variance and Averaged Correlations ---")
        #for i in range(len(features)):

            # upper triangle, avoid self-correlation and duplicates
            #for j in range(i + 1, len(features)):
                #mean_corr = mean_corr_df.iat[i, j]
                #var_corr = var_df.iat[i, j]

                #if abs(mean_corr) > 0.8 and var_corr > 0.1:
                    #print(f"{features[i]} ↔ {features[j]} → {var_corr:.2f} | {mean_corr:.2f}")

        #break

def get_s1_features(path : Path):
    return pd.read_csv(path / 's1_general_qs.csv')

def get_s2_features(path : Path):
    df = pd.DataFrame()

    # iterate through hosts and acquire respective s2s
    for entry in os.listdir(path):
        entry_path = path / entry

        if entry_path.is_dir():
            row = pd.read_csv(entry_path / 's2_selected_qs.csv')
            df = pd.concat([df, row], ignore_index=True)

    return df

def get_s3_features(path : Path):
    df = pd.DataFrame()

    # iterate through hosts and acquire respective s3s
    for entry in os.listdir(path):
        entry_path = path / entry

        if entry_path.is_dir():
            row = pd.read_csv(entry_path / 's3_connection_qs.csv')
            df = pd.concat([df, row], ignore_index=True)

    return df

def get_connection_features(path : Path):
    df = pd.DataFrame()

    # iterate over hosts
    for entry in os.listdir(path):
        entry_path = path / entry

        if entry_path.is_dir():
            connections_path = entry_path / 'connections'

            # iterate over connections
            for connection in os.listdir(connections_path):
                connection_path = connections_path / connection

                # iterate over ten second windows and acquire connection features
                for c_entry in os.listdir(connection_path):
                    connection_feature = pd.read_csv(connection_path / c_entry / 'host_data_chunk_full.csv')
                    df = pd.concat([df, connection_feature])
    
    return df

# dummy
def feed_to_model(df : pd.DataFrame):
    print(df.iloc[0][0])

def load_windows_one_by_one(data_path : Path):

    # yield one feature df by one, instead of returning one huge aggregated df
    for window in os.listdir(data_path):
        window_path = data_path / window
        yield get_connection_features(window_path)
