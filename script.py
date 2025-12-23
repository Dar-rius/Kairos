import pandas as pd 

norm_price = pd.read_csv("./data_off/btc-usd.csv")
df_metric = pd.read_csv("./data_off/metric_market.csv")
regime_mapping = {
        "Stable": 0,
        "Volatility": 1,
        "Crisis": 2
        }

# For norm_price
# Unit Test 
close_test = norm_price["Close"].loc[0:1200]
norm_price_test = norm_price.loc[0:1200]
close_test.to_csv("./data_off/unit_test/price_close.csv")
norm_price_test.to_csv("./data_off/unit_test/norm_price.csv")
# Train Test
norm_price["Close"].to_csv("./data_off/train_test/price_close.csv")
norm_price =  norm_price.drop(["Datetime_utc","Open","High","Low","Close","Volume"], axis = 1)
norm_price.to_csv("./data_off/train_test/norm_price.csv")

# For df_metric
# Unit Test
metric_test = df_metric.loc[0:50]
metric_test.to_csv("./data_off/unit_test/metric.csv")
state_test = df_metric["state"].loc[0:50]
state_test.to_csv("./data_off/unit_test/state.csv")
# Train Test
df_metric["state"] = df_metric["state"].map(regime_mapping)
df_metric["state"] = df_metric["state"].astype(int)
df_metric["state"].to_csv("./data_off/train_test/state.csv")
df_metric = df_metric.drop(["date", "state"], axis=1)
df_metric.to_csv("./data_off/train_test/metric.csv")
