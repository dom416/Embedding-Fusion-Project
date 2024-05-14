import pandas as pd
import numpy as np
import os
import datasets

def find_valid_shape(dataset):
    for data in dataset:
        if data['embedding'] is not None:
            return np.frombuffer(data['embedding'], dtype=np.float32).reshape(data['embedding_shape']).shape
    return (1, 14, 14, 2048)  # Default shape if no valid embedding is found

def save_data_by_case_id(config_name):
    # Load the dataset for the given configuration
    dataset = datasets.load_dataset("Lab-Rasool/TCGA-LUAD", config_name, split="train")

    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset)

    valid_shape = find_valid_shape(dataset)

    # Determine the case_id field based on config
    if config_name == 'clinical_data':
        case_id_field = 'case_id'
    else:
        case_id_field = 'gdc_case_id'
   
    # Sort the DataFrame by case_id
    df.sort_values(case_id_field, inplace=True)

    # Define the base directory for saving embeddings
    base_directory = f"/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/data/"

    # Ensure the base directory exists
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    # Group by case_id and save each embedding in the corresponding folder
    grouped = df.groupby(case_id_field)
    for case_id, group in grouped:
        case_directory = os.path.join(base_directory, str(case_id))
        if not os.path.exists(case_directory):
            os.makedirs(case_directory)
        
        # Iterate through each entry in the group
        for idx, row in group.iterrows():
            if row['embedding'] is None:
                embedding = np.zeros(valid_shape, dtype=np.float32)
            else:
                embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                embedding = embedding.reshape(row['embedding_shape'])

            # Save each embedding with an index to differentiate
            file_path = os.path.join(case_directory, f"{config_name}_{idx}.npy")
            np.save(file_path, embedding)
            print(f"Saved {config_name} embedding to {file_path}")

# Example usage
save_data_by_case_id('clinical_data')
save_data_by_case_id('ct')
save_data_by_case_id('pathology_report')
save_data_by_case_id('remedis_slide_image')
save_data_by_case_id('uni_slide_image')
