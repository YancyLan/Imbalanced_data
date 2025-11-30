import pandas as pd
from hyperimpute.utils.benchmarks import simulate_scenarios
from typing import Optional, Union, Dict, List
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path

ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


@dataclass(frozen=False)
class Dataset():
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_sizes(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])


def get_raw_data(data_path: str = "data/unbalance_data.csv"):
    """
    Load the unbalanced dataset from CSV file.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        X: Feature DataFrame (numeric columns only)
        Y: Target DataFrame (attack_cat column)
    """
    df = pd.read_csv(data_path)
    
    # Keep only numeric columns for features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols]
    
    # Use attack_cat as the target
    Y = df[['attack_cat']]
    
    return X, Y


def make_dataset(args):
    name = args.dataset
    missing_ratio = args.missing_ratio
    scenario = args.scenario

    X_raw, y = get_raw_data(name)
    imputation_scenarios = simulate_scenarios(X_raw, sample_columns=False) 
    ## WAS TRUE BY DEFAULT, MEANING NOT ALL COLS GOT THIS TRT
    x_gt, x_miss, miss_mask = imputation_scenarios[scenario][missing_ratio]
    x_gt, x_miss, miss_mask = x_gt.to_numpy(), x_miss.to_numpy(), miss_mask.to_numpy()

    X_num = {}
    X_num['x_miss'] = x_miss
    X_num['x_gt'] = x_gt
    X_num['miss_mask'] = miss_mask

    D = Dataset(X_num, None)

    # zero centered ADDED: STANDARDIZE
    dataset_mean_np = np.mean(X_raw.values, axis=0, keepdims=True)
    data_std = np.std(X_raw.values, axis=0, keepdims=True)
    # test removing standardization
    # D.X_num['x_miss'] = (D.X_num['x_miss'] - dataset_mean_np) / data_std
    # D.X_num['x_gt'] = (D.X_num['x_gt'] - dataset_mean_np) / data_std

    return D, y, dataset_mean_np, data_std

def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    """Get the number of unique categories for each categorical feature."""
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def prepare_fast_dataloader(D: Dataset, split: str, batch_size: int):
    """
    Prepare a fast dataloader for training/evaluation.
    
    Args:
        D: Dataset object
        split: 'train' or 'test' (determines if shuffling is enabled)
        batch_size: Batch size for the dataloader
        
    Yields:
        Batches of (X, mask) tensors
    """
    X = torch.from_numpy(D.X_num['x_miss']).float()
    X = torch.nan_to_num(X, nan=-1)
    mask = torch.from_numpy(D.X_num['miss_mask']).float()

    dataloader = FastTensorDataLoader(X, mask, batch_size=batch_size, shuffle=(split == 'train'))
    while True:
        yield from dataloader


if __name__ == "__main__":
    # Example usage
    print("Loading unbalanced dataset...")
    X, Y = get_raw_data()
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {Y.shape}")
    print(f"Attack categories: {Y['attack_cat'].unique()}")
    
    print("\nCreating dataset with missing values...")
    D = make_dataset(missing_ratio=0.3, scenario='MAR')
    print(f"x_miss shape: {D.X_num['x_miss'].shape}")
    print(f"x_gt shape: {D.X_num['x_gt'].shape}")
    print(f"miss_mask shape: {D.X_num['miss_mask'].shape}")
    print(f"Missing ratio: {D.X_num['miss_mask'].mean():.4f}")
