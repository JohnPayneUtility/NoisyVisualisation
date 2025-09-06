import pandas as pd
import pickle
import os

def save_or_append_results(df, filename):
    """
    Save DataFrame to pickle file, appending if file exists.
    """
    if os.path.exists(filename):
        # Load existing data
        with open(filename, 'rb') as f:
            existing_df = pickle.load(f)
        # Append new data
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df
    
    # Save combined data
    with open(filename, 'wb') as f:
        pickle.dump(combined_df, f)

def convert_to_split_edges_format(lon_data):
    """
    Convert LON data to split edges format.
    """
    # This is a placeholder implementation
    # You may need to implement the actual conversion logic based on your needs
    return lon_data
