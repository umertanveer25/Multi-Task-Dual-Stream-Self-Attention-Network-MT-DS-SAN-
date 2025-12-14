import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", 
    "logged_in", "num_compromised", "root_shell", "su_attempted", 
    "num_root", "num_file_creations", "num_shells", "num_access_files", 
    "num_outbound_cmds", "is_host_login", "is_guest_login", 
    "count", "srv_count", "serror_rate", "srv_serror_rate", 
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
    "dst_host_serror_rate", "dst_host_srv_serror_rate", 
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", 
    "label", "difficulty_level"
]
CAT_COLS = ["protocol_type", "service", "flag"]
CONT_COLS = [c for c in COLUMNS if c not in CAT_COLS + ["label", "difficulty_level"]]

def diagnose():
    print("Reading Train CSV...")
    s = time.time()
    train_df = pd.read_csv("../data/KDDTrain+.txt", names=COLUMNS)
    print(f"Read {len(train_df)} rows in {time.time() - s:.2f}s")

    print(f"Cat Cols: {train_df[CAT_COLS].nunique()}")
    
    print("Fitting OneHotEncoder...")
    s = time.time()
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    enc.fit(train_df[CAT_COLS])
    print(f"Fitted in {time.time() - s:.2f}s")
    
    print("Transforming...")
    s = time.time()
    res = enc.transform(train_df[CAT_COLS])
    print(f"Transformed to shape {res.shape} in {time.time() - s:.2f}s")

if __name__ == "__main__":
    diagnose()
