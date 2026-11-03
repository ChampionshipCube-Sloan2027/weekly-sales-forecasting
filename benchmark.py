import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

train = pd.read_csv("store-sales-time-series-forecasting/train.csv", parse_dates=["date"])

weekly = (train.groupby(pd.Grouper(key="date", freq="W-SUN"))["sales"]
          .sum().rename("y").to_frame())
weekly = weekly.iloc[1:-1]  # drop the partial first and last weeks

df = weekly.copy()
# Every feature uses past values only. shift(1) stops the target leaking in.
for lag in [1, 2, 4, 52]:
    df[f"lag_{lag}"] = df["y"].shift(lag)
df["roll_4_mean"] = df["y"].shift(1).rolling(4).mean()
df["week_of_year"] = df.index.isocalendar().week.astype(int)
df["month"] = df.index.month
df = df.dropna()

# Target is the week-on-week change, not the sales level.
df["diff"] = df["y"] - df["lag_1"]

# Hindsight flag: 1 for the weeks ending 17 April to 1 May 2016, around the earthquake.
# This uses knowledge of the event after it happened, so it is an upper bound, not a real forecast.
df["quake_flag"] = 0
df.loc[(df.index >= "2016-04-17") & (df.index <= "2016-05-01"), "quake_flag"] = 1

X_base = df.drop(columns=["y", "diff", "quake_flag"])
X_flag = df.drop(columns=["y", "diff"])
y = df["y"]
y_diff = df["diff"]

results = []
for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=5).split(X_base), start=1):
    yt = y.iloc[te]
    print(f"fold {fold} test: {yt.index[0].date()} to {yt.index[-1].date()}")

    row = {"fold": fold}
    for name, X in [("base", X_base), ("flag", X_flag)]:
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4)
        model.fit(X.iloc[tr], y_diff.iloc[tr])
        pred = X["lag_1"].iloc[te].values + model.predict(X.iloc[te])
        row[f"xgb_{name}_rmse"] = np.sqrt(mean_squared_error(yt, pred))
        row[f"xgb_{name}_r2"] = r2_score(yt, pred)

    naive52 = X_base["lag_52"].iloc[te]
    naive1 = X_base["lag_1"].iloc[te]
    row["naive52_rmse"] = np.sqrt(mean_squared_error(yt, naive52))
    row["naive1_rmse"] = np.sqrt(mean_squared_error(yt, naive1))
    row["naive52_r2"] = r2_score(yt, naive52)
    row["naive1_r2"] = r2_score(yt, naive1)
    results.append(row)

out = pd.DataFrame(results)
print(out.round(2))
print("\nMean across folds:")
print(out.mean(numeric_only=True).round(2))