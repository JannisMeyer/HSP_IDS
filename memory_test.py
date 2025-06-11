import data_analysis.data_processing as dp
import data_analysis.data_learning as dl
import os, psutil, gc
import pyarrow.parquet as pq
from pyarrow import float32, float64, bool_

# reload module to bypass caching
import importlib
importlib.reload(dp)
importlib.reload(dl)

def read_parquet(path):
    schema = pq.read_schema(path)
    all_cols = schema.names
    all_types = [schema.field(i).type for i in range(len(schema))]

    selected_columns = [
        col for col, typ in zip(all_cols[7:], all_types[7:]) 
        if typ in (float32(), float64(), bool_())
    ]

    return dp.pd.read_parquet(path, columns=selected_columns)


# paths
test_window_path_home = dp.Path(r'\\?\C:\Users\jannis\Documents\HSP_IDS\Material\Aktuell\2025-02-17_11-14-33_192.168.1.0-normal_1\1554220324.748197-1554220354.748197') # treat it as a long path to avoid path length issues on windows
test_window_path_remote = dp.Path(r'/home/hsp252/nas_mount/hunter.ids.data/hunter.ids.preprocessor/processed_dataframes/angriff/2025-03-04_00-03-20_192.168.1.0-normal_DDoS_1/1556203726.876922-1556203756.876922')

data_set_path = dp.Path(r'/home/hsp252/nas_mount/hunter.ids.data/hunter.ids.preprocessor/processed_dataframes')
ddos_test_path_parquet = dp.Path(r'/home/hsp252/Development/DDoS')

# getconnection-based fvs from parquet
print("Before loading:")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)

ddos_fvs = read_parquet('/home/hsp252/Development/intrusion_sorted/DDoS/intrusion_normal_DDoS_10_final.parquet')
#gc.collect()
dos_fvs = read_parquet('/home/hsp252/Development/intrusion_sorted/DoS/intrusion_normal_DoS_1_final.parquet')
#gc.collect()
injection_fvs = read_parquet('/home/hsp252/Development/intrusion_sorted/injection/intrusion_injection_normal1_final.parquet')
mitm_fvs = read_parquet('/home/hsp252/Development/intrusion_sorted/mitm/intrusion_MITM_normal1_final.parquet')
runsomeware_fvs = read_parquet('/home/hsp252/Development/intrusion_sorted/runsomware/intrusion_normal_runsomware1_final.parquet')
scanning_fvs = read_parquet('/home/hsp252/Development/intrusion_sorted/scanning/intrusion_normal_scanning1_final.parquet') # not in CSVs!
normal_fvs = read_parquet('/home/hsp252/Development/benign/benign_normal_10_final.parquet')

print("After loading:")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)

del ddos_fvs
del dos_fvs
del injection_fvs
del mitm_fvs
del runsomeware_fvs
del scanning_fvs
del normal_fvs
gc.collect()

print("After deleting:")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)

while True:
    pass