import pandas as pd 

norm_price = pd.read_csv("./data_off/btc-usd.csv")
df_metric = pd.read_csv("./data_off/metric_market.csv")
#train test
price_train = pd.read_csv("./data_off/train_test/price_train.csv")
metric_train = pd.read_csv("./data_off/train_test/metric_train.csv")
price_test = pd.read_csv("./data_off/train_test/price_test.csv")
metric_test = pd.read_csv("./data_off/train_test/metric_test.csv")
#Transform str to int in state variable
regime_mapping = {
        "Stable": 0,
        "Volatility": 1,
        "Crisis": 2
        }
for dataset in [df_metric, metric_train, metric_test]:
    dataset["state"] = dataset["state"].map(regime_mapping)
    dataset["state"] = dataset["state"].astype(int)

# For norm_price
# Unit Test
close_unit_test = norm_price["Close"].loc[0:1200]
norm_price_unit_test = norm_price.loc[0:1200]
close_unit_test.to_csv("./data_off/unit_test/price_close.csv")
norm_price_unit_test.to_csv("./data_off/unit_test/norm_price.csv")
metric_unit_test = df_metric.loc[0:50]
metric_unit_test.to_csv("./data_off/unit_test/metric.csv")
state_unit_test = df_metric["state"].loc[0:50]
state_unit_test.to_csv("./data_off/unit_test/state.csv")

# Train for norm price
price_train["Close"].to_csv("./data_off/train_test/price_close_train.csv")
price_train =  norm_price.drop(["Datetime_utc","Open","High","Low","Close","Volume"], axis = 1)
price_train.to_csv("./data_off/train_test/price_train.csv")
# Test for norm price
price_test["Close"].to_csv("./data_off/train_test/price_close_test.csv")
price_test =  norm_price.drop(["Datetime_utc","Open","High","Low","Close","Volume"], axis = 1)
price_test.to_csv("./data_off/train_test/price_test.csv")

# Train form metric
metric_train["state"].to_csv("./data_off/train_test/state_train.csv")
metric_train = metric_train.drop(["date", "state"], axis=1)
metric_train.to_csv("./data_off/train_test/metric_train.csv")
# Test form metric
metric_test["state"].to_csv("./data_off/train_test/state_test.csv")
metric_test = metric_test.drop(["date", "state"], axis=1)
metric_test.to_csv("./data_off/train_test/metric_train.csv")
