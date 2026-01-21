import pandas as pd
from torch import Tensor
import torch

def to_df(new_df: dict) -> pd.DataFrame:
    df = pd.DataFrame(new_df)
    return df

def concat_df(old_df: pd.DataFrame, new_df: list[int, float]) -> pd.DataFrame:
    old_df.loc[len(old_df)] = new_df
    return old_df

def convert_and_save_df(data: dict, filename: str):
    df = to_df(data)
    df.to_csv(filename, orient = 'index')

#Convert the Tensor to List
def convert_tensor_to_list(tensor: Tensor): return tensor.tolist()

def convert_to_btc(amount_usd: Tensor, btc_value: Tensor, device:str): return amount_usd / btc_value

def convert_to_usd(amount_btc: Tensor, btc_value: Tensor, device:str): return amount_btc * btc_value

#Convert df to list
def df_to_list(dataset: pd.DataFrame) -> tuple:
    open_ = dataset["Open"].to_list()
    high_ = dataset["High"].to_list()
    low_ = dataset["Low"].to_list()
    close_ = dataset["Close"].to_list()
    volume_ = dataset["Volume"].to_list()
    return (open_, high_, low_, close_, volume_)
