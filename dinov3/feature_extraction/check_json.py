import json
import glob
import os
import random

def check_json_structure(input_dir):
    print(f"Searching for JSONs in {input_dir}...")
    files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)
    
    if not files:
        print("No JSON files found.")
        return

    print(f"Found {len(files)} files. Sampling 3...")
    samples = random.sample(files, min(3, len(files)))
    
    for fpath in samples:
        print(f"\n[Checking] {fpath}")
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            print(f"  Type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())[:5]}")
                if 'images' in data:
                    print(f"  Has 'images' key. First item: {data['images'][0]}")
                else:
                    # Check first item structure
                    first_key = next(iter(data))
                    first_val = data[first_key]
                    print(f"  First Key (Path?): {first_key}")
                    print(f"  First Value Type: {type(first_val)}")
                    print(f"  First Value Content: {first_val}")
            elif isinstance(data, list):
                print(f"  List length: {len(data)}")
                if data:
                    print(f"  First item: {data[0]}")
                    
        except Exception as e:
            print(f"  Error reading {fpath}: {e}")

if __name__ == "__main__":
    # You can change this path or pass it as arg
    # Defaulting to the path you provided earlier
    check_json_structure("/mnt/huawei_deepcad/onepb/shards_task2/standard/small_square/")

