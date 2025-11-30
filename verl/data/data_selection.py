import json
import pandas as pd
import random 
import numpy as np
import os

import argparse

def compute_score(input_file, alpha=0):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    scores = {}
    try:
        alpha = float(alpha)
    except ValueError:
        print(f"Invalid alpha value: {alpha}, using default 0.5")
        alpha = 0.5

    # print(f"Computing scores with alpha={alpha}")
    # if data:
    #     first_key = next(iter(data))
    #     print(f"Sample data structure (key={first_key}): {data[first_key]}")

    if isinstance(data, dict):
        if data :
            first_key = next(iter(data))
            print(f"Sample data structure (key={first_key}): {data[first_key]}")   
        iterable= data.items()
    elif isinstance(data, list):
        if data :
            print(f"Sample data structure (first item): {data[0]}")   
        iterable= enumerate(data)
        def gen_from_list(data):
            for i, item in enumerate(data):
                if isinstance(item, dict) and 'idx' in item:
                    yield str(item['idx']), item
                else: 
                    yield str(i), item
        iterable = gen_from_list(data)

    for key, value in iterable:
        score_norm =0
        
        # entropy = 0
        # eval_score = 0
        if isinstance(value, dict):
            try:
                score_norm = float(value.get('score_norm', 0))
            except Exception:
                score_norm = 0
        elif isinstance(value, list) and len(value) >= 1:
            try:
                score_norm = float(value[0])
            except Exception:
                score_norm = 0
        final_score = score_norm
        #     try:
        #         entropy = float (value.get('entropy', 0))
        #     except Exception:
        #         entropy = 0
        #     try:
        #         eval_score = float (value.get('eval', 0))
        #     except Exception:
        #         eval_score = 0
        # elif isinstance(value, list) and len(value) >= 2:
        #     try:
        #         entropy = float (value[0])
        #     except Exception:
        #         entropy = 0
        #     try:
        #         eval_score = float (value[1])
        #     except Exception:
        #         eval_score = 0
        
        # # Compute weighted score: alpha * entropy + (1 - alpha) * eval
        # final_score = alpha * entropy + (1 - alpha) * eval_score
        print(final_score)
        scores[str(key)] = -final_score

    # Sort by score in descending order
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_scores


def sample_parquet_by_indices(json_file_path, parquet_file_path, output_parquet_path, 
                            index_column='index', top_n=128, top_start=None, top_end=None, 
                            alpha=0, repeat_time=1, is_save=1):
    if top_start is not None and top_end is not None:
        print(f"Sampling records from index {top_start} to {top_end} from {parquet_file_path} using alpha={alpha}.")
        expected_sample_count = (top_end - top_start + 1)
    else:
        print(f"Sampling {top_n} records from {parquet_file_path} using alpha={alpha}.")
        expected_sample_count = top_n
        
    sorted_counts = compute_score(json_file_path, alpha)
    if not sorted_counts:
        print("No valid counts obtained from the JSON file / parquet file.")
        return 0, 0
    
    # Load the original JSON data to access raw values
    with open(json_file_path, 'r') as f:
        original_json_data=  {}
        for line in f:
            if line.strip():
                item = json.loads(line)
                idx = item.get('idx', None)
                if idx is not None:
                    original_json_data[str(idx)] = item
            

    string_indices = list(sorted_counts.keys())
    
    # Select indices based on range if provided, otherwise use top_n
    if top_start is not None and top_end is not None:
        string_indices = string_indices[top_start:top_end+1]
    else:
        string_indices = string_indices[:min(top_n, len(sorted_counts))]
    # transform string indices to int
    indices_to_keep = [int(idx) for idx in string_indices]
    print(indices_to_keep)
    # origin data path to select specific indices
    df = pd.read_parquet(parquet_file_path, engine='pyarrow')

    result_data = []
    # _ represent ignore the index if index not in indices_to_keep
    for _, row in df.iterrows():
        try:
            if row['extra_info']['index'] in indices_to_keep:
                result_data.append(row)
        except:
            continue
    #transfer to the dataframe form
    filtered_df = pd.DataFrame(result_data)
    
    # Handle data repetition if repeat_time > 1
    if repeat_time > 1:
        # Create a list to store the repeated dataframes
        repeated_dfs = [filtered_df] * repeat_time
        # Concatenate all the repeated dataframes
        filtered_df = pd.concat(repeated_dfs, ignore_index=True)
        
        # Update the output path to include the repeat information
        total_count = expected_sample_count * repeat_time
        output_base, output_ext = os.path.splitext(output_parquet_path)
        output_parquet_path = f"{output_base}_repeatto{total_count}{output_ext}"
    
    # Randomly select up to 10 samples from the filtered dataframe
    sample_size = min(10, len(filtered_df))
    sample_indices = random.sample(range(len(filtered_df)), sample_size)
    print("\n=== Randomly selected samples ===")
    
    # for i, idx in enumerate(sample_indices):
    #     sample_row = filtered_df.iloc[idx]
    #     row_index = str(sample_row['extra_info']['index'])
        
    #     print(f"\nSample {i+1} (Index: {row_index}):")
    #     print("Raw count list from JSON:")
    #     print(original_json_data.get(row_index, []))
    #     print("Sample row data:")
    #     print(sample_row)
    #     print("prompt:")
    #     print(sample_row['prompt'][0]['content'])
    #     print("-" * 80)
    
    # Count unique prompts for verification
    unique_prompts = set()
    for _, row in filtered_df.iterrows():
        prompt_content = row['prompt'][0]['content']
        unique_prompts.add(prompt_content)
    
    print(f"Sampled {len(filtered_df)} records out of {len(df)} total.")
    print(f"Number of unique prompts in the output file: {len(unique_prompts)}")
    
    # Save to new parquet file
    if is_save:
        filtered_df.to_parquet(f"{output_parquet_path}")
        print(f"Saved to {output_parquet_path}")
    else:
        print(f"Not saving to file (is_save=0). Would have saved to {output_parquet_path}")
    
    return len(filtered_df), len(unique_prompts)

def random_sample_parquet(parquet_file_path, output_parquet_path, sample_size=128, is_save=1):
    df = pd.read_parquet(parquet_file_path, engine='pyarrow')
    
    total_records = len(df)
    
    if total_records <= sample_size:
        print(f"Warning: Requested sample size ({sample_size}) is greater than or equal to "
                f"the total number of records ({total_records}). Returning all records.")
        df.to_parquet(output_parquet_path)
        return total_records
    
    random_indices = random.sample(range(total_records), sample_size)
    
    sampled_df = df.iloc[random_indices]
    
    # Save to new parquet file
    if is_save:
        sampled_df.to_parquet(output_parquet_path)
        print(f"Saved to {output_parquet_path}")
    else:
        print(f"Not saving to file (is_save=0). Would have saved to {output_parquet_path}")
    
    return sample_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample parquet files based on different methods.')
    parser.add_argument("--input_path", type=str, default="/root/autodl-tmp/TTRL/verl/data/answer.jsonl", help="Path to the index json file")
    parser.add_argument("--data_dir", type=str, default="train/one_shot_ttrl", help="Path to the data directory")
    parser.add_argument("--parquet_file_name", type=str, default="/root/autodl-tmp/TTRL/verl/data/MATH-TTT/train.parquet", help="Path to the parquet file")
    parser.add_argument('--top_n', type=int, default=2, help='Number of top records to sample')
    parser.add_argument('--repeat_time', type=int, default=32, help='Number of times to repeat the sampling')
    parser.add_argument('--top_index', type=int, default=None, help='Index of the top record')
    parser.add_argument('--top_start', type=int, default=1, help='Start index of the range selection')
    parser.add_argument('--top_end', type=int, default=1, help='End index of the range selection')
    parser.add_argument('--alpha', type=str, default='0.4', help='Method to sample the parquet file')
    parser.add_argument('--is_save', type=int, default=1, help='Whether to save the output parquet file (1=yes, 0=no)')
    args = parser.parse_args()


    input_path = args.input_path
    data_dir = args.data_dir
    parquet_file_path = args.parquet_file_name

    top_index = args.top_index
    top_n = args.top_n 
    repeat_time = args.repeat_time
    # top_index = args.top_index
    top_start = args.top_start  # Default to None when not using range selection
    top_end = args.top_end    # Default to None when not using range selection
    if top_index is not None:
        top_start = top_index
        top_end = top_index
    print(f"top_start: {top_start}, top_end: {top_end}, top_n: {top_n}")


    alpha = args.alpha
    is_save = args.is_save

    if top_start is not None and top_end is not None:
        if top_start == top_end:
            output_parquet_path = f"{input_path}_{alpha}_pi{top_start+1}_r{repeat_time}.parquet"
        else:
            output_parquet_path = f"{input_path}_{alpha}_pi{top_start+1}-{top_end+1}_r{repeat_time}.parquet"
    else:
        output_parquet_path = f"{input_path}_{alpha}_sample{top_n}_r{repeat_time}.parquet"
        
    # print(acc_score(index_json_path, method=alpha))
    sample_parquet_by_indices(input_path, parquet_file_path, output_parquet_path, 
                            top_n=top_n, top_start=top_start, top_end=top_end,
                            alpha=alpha, repeat_time=repeat_time, is_save=is_save)
    # random_sample_parquet(parquet_file_path, output_parquet_path.replace('.parquet', '_random.parquet'), sample_size=top_n, is_save=is_save)
