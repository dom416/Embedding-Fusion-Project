import pandas as pd
import numpy as np
import os
import datasets

def save_data_by_case_id(config_name):
    # Load the dataset for the given configuration
    dataset = datasets.load_dataset("Lab-Rasool/TCGA-LUAD", config_name, split="train")

    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset)

    # Determine the case_id field based on config
    if config_name == 'clinical_data':
        case_id_field = 'case_id'
    else:
        case_id_field = 'gdc_case_id'

    # Define the base directory for saving data
    base_directory = f"/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/data/"

    # Ensure the base directory exists
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    # Process each case based on case_id
    grouped = df.groupby(case_id_field)
    for case_id, group in grouped:
        case_directory = os.path.join(base_directory, str(case_id))
        if not os.path.exists(case_directory):
            os.makedirs(case_directory)
        
        # Iterate through each entry in the group
        for idx, row in group.iterrows():
            # Check if days_to_death is null
            if pd.isnull(row['days_to_death']):
                censor = 1
                survival_time = row['days_to_last_follow_up']
            else:
                censor = 0
                survival_time = row['days_to_death']

            # Save censor and survival_time to the folder
            with open(os.path.join(case_directory, 'censor.npy'), 'w') as f:
                f.write(censor)
            with open(os.path.join(case_directory, 'survival_time.npy'), 'w') as f:
                f.write(str(survival_time))

            print(f"Saved censor and survival time for case_id {case_id} at index {idx}")

# Example usage
save_data_by_case_id('clinical_data')
