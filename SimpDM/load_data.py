import pandas as pd
from hyperimpute.utils.benchmarks import simulate_scenarios
import json
from typing import Optional, Union, Dict, List
import numpy as np
import torch
from dataclasses import dataclass

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
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])@dataclass(frozen=False)

def get_raw_data(dset_name):

    if dset_name == 'iris':
        from sklearn.datasets import load_iris
        X, Y = load_iris(as_frame=True, return_X_y=True)

    elif dset_name == 'wine_white':
        X, Y = fetch_wine_quality_white()

    elif dset_name == 'airfoil':
        X, Y = fetch_airfoil()

    elif dset_name == 'yeast':
        X, Y = fetch_yeast()

    elif dset_name == 'california':
        from sklearn.datasets import fetch_california_housing
        X, Y = fetch_california_housing(as_frame=True, return_X_y=True)

    elif dset_name in ["yacht", "housing", "diabetes", "blood", "energy", "german", "concrete",
                                "wine_red", "abalone", "phoneme", "power", "ecommerce"]:
        df_np = np.loadtxt('./raw_data/{}/data/data.txt'.format(dset_name))
        X = pd.DataFrame(df_np[:, :-1])
        Y = pd.DataFrame(df_np[:, -1:])
    
    elif dset_name == 'unbalanced':
        df = pd.read_csv('./raw_data/unbalance/unbalanced_data_standardized.csv')
        X = pd.DataFrame(df.values[:, :-2].astype('float'))
        Y = pd.DataFrame(df.values[:, -1])

    else:
        assert "error dataset name"

    return X, Y

def make_dataset(args):
    name = args.dataset
    missing_ratio = args.missing_ratio
    scenario = args.scenario

    X_raw, y = get_raw_data(name)
    print(X_raw.columns)
    imputation_scenarios = simulate_scenarios(X_raw, 
                                              column_limit=len(X_raw.columns), 
                                              sample_columns=False) 
    ##CHANGED!!! sample_columns WAS TRUE BY DEFAULT, MEANING NOT ALL COLS GOT THIS TRT
    ##ALSO, column_limit is 8 BY DEFAULT, meaning ONLY EIGHT COLS GET IT ANYWAY!!
    # you must change this to what i have for it to work properly. (missing randomly across ALL COLS)
    # i can't believe the authors didn't check this, maybe they just never ran stuff on data w more than 8 cols?
    x_gt, x_miss, miss_mask = imputation_scenarios[scenario][missing_ratio]
    x_gt, x_miss, miss_mask = x_gt.to_numpy(), x_miss.to_numpy(), miss_mask.to_numpy()

    X_num = {}
    X_num['x_miss'] = x_miss
    X_num['x_gt'] = x_gt
    X_num['miss_mask'] = miss_mask

    D = Dataset(X_num, None)

    # saving mean and std for later
    dataset_mean_np = np.mean(X_raw.values, axis=0, keepdims=True)
    data_std = np.std(X_raw.values, axis=0, keepdims=True)
    # test removing standardization
    # D.X_num['x_miss'] = (D.X_num['x_miss'] - dataset_mean_np) / data_std
    # D.X_num['x_gt'] = (D.X_num['x_gt'] - dataset_mean_np) / data_std

    # sanity check: do all columns get missing values?
    # x_miss_df = pd.DataFrame(x_miss)
    # missing_fraction = x_miss_df.isna().mean()
    # print("Fraction of missing values per column:")
    # print(missing_fraction)
    
    # Optional: overall missingness
    # overall_missing = np.isnan(x_miss).mean()
    # print(f"\nOverall fraction of missing values: {overall_missing:.3f}")
    
    return D, y, dataset_mean_np, data_std

def fetch_wine_quality_white():
    with open('./raw_data/wine_quality_white/winequality-white.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        X = pd.DataFrame(df.values[:, :-1].astype('float'))
        Y = pd.DataFrame(df.values[:, -1])
    return X, Y

def fetch_airfoil():
    with open('./raw_data/airfoil/airfoil_self_noise.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        X = pd.DataFrame(df.values[:, :-1])
        Y = pd.DataFrame(df.values[:, -1])
    return X, Y

def fetch_yeast():
    with open('./raw_data/yeast/yeast.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        X = pd.DataFrame(df.values[:, 1:-1].astype('float'))
        Y = pd.DataFrame(df.values[:, -1])
    return X, Y


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
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

def prepare_fast_dataloader(D : Dataset, split : str, batch_size: int):

    X = torch.from_numpy(D.X_num['x_miss']).float()
    X = torch.nan_to_num(X, nan=-1)
    mask = torch.from_numpy(D.X_num['miss_mask']).float()

    dataloader = FastTensorDataLoader(X, mask, batch_size=batch_size, shuffle=(split == 'train'))
    while True:
        yield from dataloader
