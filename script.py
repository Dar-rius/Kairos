import pandas as pd 

norm_price = pd.read_csv("./data_off/btc-usd.csv")
df_metric = pd.read_csv("./data_off/metric_market.csv")

norm_price = norm_price.loc[0:50]
norm_price["Close"].to_csv("./data_off/unit_test/unit_test_price_close.csv")
norm_price =  norm_price.drop(["Datetime_utc","Open","High","Low","Close","Volume"], axis = 1)
norm_price.to_csv("./data_off/unit_test/unit_test_norm_price.csv")
df_metric = df_metric.drop(["date", "state"], axis=1)
df_metric.loc[0:50].to_csv("./data_off/unit_test/unit_test_df_metric.csv")
