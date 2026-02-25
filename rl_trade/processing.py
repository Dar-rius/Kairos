from torch import Tensor
import torch

#Convert the Tensor to List
def tensor_to_list(tensor: Tensor) -> list: return tensor.tolist()

def convert_to_btc(amount_usd: Tensor, btc_value: Tensor) -> Tensor: return amount_usd / btc_value

def convert_to_usd(amount_btc: Tensor, btc_value: Tensor) -> Tensor: return amount_btc * btc_value
