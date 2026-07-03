import pandas as pd 
import os


#This script process all dataset
PATH_EXIST = "./data_off/train_test/price_close_train.csv"
if not os.path.exists(PATH_EXIST):
    #Load dataset
    market_metric_pretrain = pd.read_csv("./data_off/train_test/metric_pretrain.csv")
    market_metric_train = pd.read_csv("./data_off/train_test/metric_train.csv")
    market_metric_test = pd.read_csv("./data_off/train_test/metric_test.csv")
    daily_train_series = pd.read_csv("./data_off/train_test/price_train.csv")
    daily_test_series = pd.read_csv("./data_off/train_test/price_test.csv")

    #Encode regime class from str to int
    regime_mapping = {
            "Stable": 0,
            "Volatility": 1,
            "Crisis": 2
            }
    for dataset in [market_metric_train, market_metric_pretrain, market_metric_test]:
        dataset["regime"] = dataset["regime"].map(regime_mapping)
        dataset["regime"] = dataset["regime"].astype(int)


    #Process micro trade dataset for train
    price_close_train = daily_train_series["Close"]
    price_close_train.to_csv("./data_off/train_test/price_close_train.csv")

    daily_train_series =  daily_train_series.drop(["Datetime_utc","Open","High","Low","Close","Volume"], axis = 1)
    daily_train_series.to_csv("./data_off/train_test/daily_train.csv")

    #Delete price_train and price_train
    os.remove("./data_off/train_test/price_train.csv")


    #Process micro trade dataset for test
    price_close_test = daily_test_series["Close"]
    price_close_test.to_csv("./data_off/train_test/price_close_test.csv")

    daily_test_series =  daily_test_series.drop(["Datetime_utc","Open","High","Low","Close","Volume"], axis = 1)
    daily_test_series.to_csv("./data_off/train_test/daily_test.csv")

    #Delete price_train and price_test
    os.remove("./data_off/train_test/price_test.csv")


    #Process macro trade dataset for pre-train
    market_regime_pretrain = market_metric_pretrain["regime"]
    market_regime_pretrain.to_csv("./data_off/train_test/regime_pretrain.csv")

    market_change_pretrain = market_metric_pretrain["change"]
    market_change_pretrain.to_csv("./data_off/train_test/change_pretrain.csv")

    market_metric_pretrain = market_metric_pretrain.drop(["date", "regime", "change"], axis=1)
    market_metric_pretrain.to_csv("./data_off/train_test/metric_pretrain.csv")


    #Process macro trade dataset for train
    market_regime_train = market_metric_train["regime"]
    market_regime_train.to_csv("./data_off/train_test/regime_train.csv")

    market_change_train = market_metric_train["change"]
    market_change_train.to_csv("./data_off/train_test/change_train.csv")

    market_metric_train = market_metric_train.drop(["date", "regime", "change"], axis=1)
    market_metric_train.to_csv("./data_off/train_test/metric_train.csv")


    #Process macro trade dataset for test
    market_regime_test = market_metric_test["regime"]
    market_regime_test.to_csv("./data_off/train_test/regime_test.csv")

    market_change_test = market_metric_test["change"]
    market_change_test.to_csv("./data_off/train_test/change_test.csv")

    market_metric_test = market_metric_test.drop(["date", "regime", "change"], axis=1)
    market_metric_test.to_csv("./data_off/train_test/metric_test.csv")

    print("Data are processed and new files are created in data_off path")
else:
    print("Data are already processed")
