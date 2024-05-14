import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import os
import torch.nn.functional as F


class SurvivalDataset(Dataset):
    def __init__(self, data_dir, fold=1, seed=41,n_folds=5, split='train'):
        self.data = self.load_data(data_dir)
        self.split_data(split, fold, n_folds, seed)

    def load_embedding(self, case_path, data_type):
      if data_type == 'uni_slide':
        # List all files matching the uni_slide pattern
        files = [f for f in os.listdir(case_path) if 'uni_slide' in f]
        embeddings = []
        for file in files:
            try:
                file_path = os.path.join(case_path, file)
                embedding = np.load(file_path)
                # Ensure the embedding is correctly shaped as [x, 1024]
                embedding = torch.tensor(embedding, dtype=torch.float32).view(-1, 1024)  # Adjust as necessary
                embeddings.append(embedding)
            except FileNotFoundError:
                continue
        
        if embeddings:
            # Concatenate along the sequence dimension (axis=0)
            embeddings = torch.cat(embeddings, dim=0)
            # Apply mean pooling to reshape from [x, 1024] to [1, 1024]
            embeddings = torch.mean(embeddings, dim=0, keepdim=True) 
            return embeddings
        else:
            return torch.zeros((1, 1024))  # Return zeros if no files are found
      else:
        # Handle other data types
        try:
            file_path = next(os.path.join(case_path, f) for f in os.listdir(case_path) if data_type in f)
            embedding = np.load(file_path)
            embedding = torch.tensor(embedding, dtype=torch.float32).view(1, -1)  # Ensure consistent shape for single files
            return embedding
        except (FileNotFoundError, StopIteration):
            return torch.zeros((1, 1024))  # Return zeros if file not found





    def load_censor(self, filepath):
      censor = np.load(filepath)
      return torch.tensor(censor, dtype=torch.float32)  # Default censor value if missing or unreadable

    def load_data(self, data_dir):
        data_entries = []
        period_length = 182.5  # One period is 365/2 days
        num_periods = 20  # Total 20 periods

        for case_id in os.listdir(data_dir):
            case_path = os.path.join(data_dir, case_id)
            survival_time_path = os.path.join(case_path, 'survival_time.txt')
            censor_path = os.path.join(case_path, 'censor.npy')  # Assuming censor.txt instead of censor.npy

            try:
                with open(survival_time_path, 'r') as f:
                    survival_time = float(f.read().strip())
                if survival_time <= 0:
                    continue  # Skip cases with survival time less than or equal to 0 days
            except (FileNotFoundError, ValueError):
                continue

            # Calculate the time bin label
            true_time_bin = min(int(survival_time // period_length), num_periods - 1)  # -1 because bins start from 0
            # if time exceeds 10 years, it is binned into the last bin (index 19).

            clinical_data = self.load_embedding(case_path, 'clinical')
            pathology_report = self.load_embedding(case_path, 'pathology')
            uni_slide_image = self.load_embedding(case_path, 'uni_slide')
            censor = self.load_censor(os.path.join(case_path, 'censor.npy'))

            data_entries.append({
                'case_id': case_id,
                'survival_time': survival_time,
                'true_time_bin': true_time_bin,
                'censor': censor,
                'clinical_data': clinical_data,
                'pathology_report': pathology_report,
                'uni_slide_image': uni_slide_image
            })

        return data_entries
  

    def split_data(self, split, fold, n_folds, seed):
        torch.manual_seed(seed)  # Set the random seed for reproducibility
        total_cases = len(self.data)
        indices = torch.randperm(total_cases).tolist()  # Shuffle indices

        # Calculate the size of each fold
        fold_size = total_cases // n_folds
        remainders = total_cases % n_folds

        # Ensure each fold gets an approximately equal number of samples
        sizes = [fold_size + (1 if i < remainders else 0) for i in range(n_folds)]

        # Determine the test indices for the current fold
        start_index = sum(sizes[:fold - 1])  # sum previous fold sizes to get start index
        end_index = start_index + sizes[fold - 1]  # add current fold size to get end index

        # Create index lists for train and test datasets
        test_indices = indices[start_index:end_index]
        train_indices = indices[:start_index] + indices[end_index:]

        # Extract train and test datasets based on indices
        self.test_data = [self.data[i] for i in test_indices]
        self.train_data = [self.data[i] for i in train_indices]

        # Set data to the appropriate split
        self.data = self.train_data if split == 'train' else self.test_data

    def __getitem__(self, idx):
        entry = self.data[idx]
        return {
            'case_id': entry['case_id'],
            'survival_time': entry['survival_time'],
            'censor': entry['censor'],
            'clinical_data': entry['clinical_data'],
            'pathology_report': entry['pathology_report'],
            'uni_slide_image': entry['uni_slide_image'],
            'true_time_bin': entry['true_time_bin']
        }

    def __len__(self):
        return len(self.data) 
