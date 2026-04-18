import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool


DATASET_NAME = "JeanKaddour/minipile"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/data"
MAX_SHARD = 14
SPLIT = "validation" 
DATA_DIR = r"C:\\Users\\Ashmit Gupta\\Desktop\\Coding\\Pytorch\\Transformer\\GPT\\Professional GPT\\Preprocess Data (Pre Train)\\data_test"
os.makedirs(DATA_DIR, exist_ok=True)


def download_single_file(filename):


    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True


    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")


    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024): 
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True


        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False


    return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1 = all)")
    parser.add_argument("-w", "--num-workers", type=int, default=8, help="Number of parallel download workers")
    parser.add_argument("-s", "--split", type=str, default="train", help="Dataset split to download (train/validation/test)")
    args = parser.parse_args()


    # Fetch file list from HuggingFace API (5 NEW LINES)
    api_url = f"https://huggingface.co/api/datasets/{DATASET_NAME}/tree/main/data"
    response = requests.get(api_url)
    files_data = response.json()
    filenames = sorted([item['path'].split('/')[-1] for item in files_data if item['path'].endswith('.parquet') and SPLIT in item['path']])
    
    files_to_download = filenames if args.num_files == -1 else filenames[:args.num_files]
    print(f"Downloading {len(files_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, files_to_download)


    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(files_to_download)} shards to {DATA_DIR}")



    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(files_to_download)} shards to {DATA_DIR}")
