import polars as pl

train_df = pl.read_csv("./input/train.csv")
print(train_df.describe())
print(train_df.head(5))
