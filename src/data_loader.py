import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

# Separate features for Dual-Stream
CAT_COLS = ["protocol_type", "service", "flag"]
# All others excluding label/difficulty are continuous
CONT_COLS = [c for c in COLUMNS if c not in CAT_COLS + ["label", "difficulty_level"]]

class NSLDataset(Dataset):
    def __init__(self, x_cont, x_cat, y):
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.float32) # OneHot is also float for processing
        # Encode label: Normal=0, Attack=1
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_cont[idx], self.x_cat[idx], self.y[idx]

def load_data(train_path, test_path, batch_size=64):
    print("Loading datasets...")
    train_df = pd.read_csv(train_path, names=COLUMNS)
    test_df = pd.read_csv(test_path, names=COLUMNS)

    # Binary Classification: Normal vs Attack
    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Preprocessing Pipelines
    # Continuous: MinMax Scaling
    # Categorical: OneHotEncoding
    
    print("Fitting preprocessors...")
    # fit on TRAIN only
    cont_scaler = MinMaxScaler()
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    x_train_cont = cont_scaler.fit_transform(train_df[CONT_COLS])
    x_train_cat = cat_encoder.fit_transform(train_df[CAT_COLS])
    y_train = train_df['label'].values

    x_test_cont = cont_scaler.transform(test_df[CONT_COLS])
    x_test_cat = cat_encoder.transform(test_df[CAT_COLS])
    y_test = test_df['label'].values

    print(f"Train shapes: Cont={x_train_cont.shape}, Cat={x_train_cat.shape}")
    print(f"Test shapes: Cont={x_test_cont.shape}, Cat={x_test_cat.shape}")

    train_dataset = NSLDataset(x_train_cont, x_train_cat, y_train)
    test_dataset = NSLDataset(x_test_cont, x_test_cat, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dims = {
        'cont': x_train_cont.shape[1],
        'cat': x_train_cat.shape[1]
    }

    return train_loader, test_loader, input_dims

if __name__ == "__main__":
    # Test the loader
    train_p = "../data/KDDTrain+.txt"
    test_p = "../data/KDDTest+.txt"
    try:
        t_loader, _, dims = load_data(train_p, test_p)
        print("Data Loader Verification Successful.")
        print(f"Input Dims: {dims}")
    except FileNotFoundError:
        print("Data files not found. Run download_data.py first.")
