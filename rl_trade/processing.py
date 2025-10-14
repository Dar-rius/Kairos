import pandas as pd

def to_df(new_df: dict) -> pd.DataFrame:
    df = pd.DataFrame(new_df)
    return df


def concat_df(old_df: dict, new_df: dict) -> pd.DataFrame:
    df = pd.concat(old_df, new_df)
    return df 

def convert_and_save_df(data: dict, filename: str):
    df = to_df(data)
    df.to_csv(filename, orient = 'index')
