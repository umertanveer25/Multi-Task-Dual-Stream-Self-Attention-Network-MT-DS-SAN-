import os
import requests
import pandas as pd

DATA_DIR = r"C:\Users\umert\.gemini\antigravity\playground\photonic-planetary\NSL_KDD_Novel_Research\data"
BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"

FILES = {
    "KDDTrain+.txt": "KDDTrain+.txt",
    "KDDTest+.txt": "KDDTest+.txt"
}

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping.")
        return
    
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)
        print("Done.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname, url_suffix in FILES.items():
        download_file(BASE_URL + url_suffix, os.path.join(DATA_DIR, fname))

    # Also adding feature names since the txt files are headerless
    # Basic NSL-KDD Columns
    columns = [
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
    
    # Save column names for reference
    with open(os.path.join(DATA_DIR, "columns.txt"), "w") as f:
        f.write("\n".join(columns))

if __name__ == "__main__":
    main()
