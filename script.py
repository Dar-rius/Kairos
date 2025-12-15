import pandas as pd 

df_price = pd.read_csv("./data_off/btc-usd_dataset.csv")
df_metric = pd.read_csv("./data_off/metric_market.csv")

df_price.loc[0:50000].to_csv("./data_off/unit_test/unit_test_df_price.csv")
df_metric.loc[0:1000].to_csv("./data_off/unit_test/unit_test_df_metric.csv")
