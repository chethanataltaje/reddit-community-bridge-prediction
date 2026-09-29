import pandas as pd
import numpy as np

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the TSV dataset (soc-redditHyperlinks-title.tsv).
    """
    # Assuming tab-separated as requested
    df = pd.read_csv(filepath, sep='\t')
    return df

def basic_validation(df: pd.DataFrame) -> dict:
    """
    Perform basic validation of the dataset.
    Returns a dictionary containing shape, columns, missing values, etc.
    """
    validation_results = {
        "shape": df.shape,
        "columns": list(df.columns)
    }
    
    validation_results["missing_values"] = df.isnull().sum().to_dict()
    validation_results["duplicate_rows"] = int(df.duplicated().sum())
    
    if 'SOURCE_SUBREDDIT' in df.columns and 'TARGET_SUBREDDIT' in df.columns:
        validation_results["unique_source"] = df['SOURCE_SUBREDDIT'].nunique()
        validation_results["unique_target"] = df['TARGET_SUBREDDIT'].nunique()
        
        all_nodes = set(df['SOURCE_SUBREDDIT']).union(set(df['TARGET_SUBREDDIT']))
        validation_results["unique_nodes"] = len(all_nodes)
        
        unique_edges = df.drop_duplicates(subset=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT'])
        validation_results["unique_edges"] = len(unique_edges)
        
        validation_results["self_loops"] = int((df['SOURCE_SUBREDDIT'] == df['TARGET_SUBREDDIT']).sum())
        
    return validation_results

import pickle
import os

def save_processed_data(G_train, train_pos, test_pos, test_neg, save_dir='../data/processed'):
    """
    Saves the processed graph and edge splits to disk to avoid recomputing.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save NetworkX Graph
    with open(os.path.join(save_dir, 'G_train.pkl'), 'wb') as f:
        pickle.dump(G_train, f)
        
    # Save edge splits as CSV
    pd.DataFrame(train_pos, columns=['source', 'target']).to_csv(os.path.join(save_dir, 'train_edges_pos.csv'), index=False)
    pd.DataFrame(test_pos, columns=['source', 'target']).to_csv(os.path.join(save_dir, 'test_edges_pos.csv'), index=False)
    pd.DataFrame(test_neg, columns=['source', 'target']).to_csv(os.path.join(save_dir, 'test_edges_neg.csv'), index=False)
    
    print(f"Successfully cached processed graph and splits to {save_dir}/")
