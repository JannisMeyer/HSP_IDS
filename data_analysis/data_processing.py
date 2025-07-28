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


# region data plotting -----------------------------------------------------------------------------------------------------------------------

USEFUL_FEATURES_START = 'conn_duration'

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

