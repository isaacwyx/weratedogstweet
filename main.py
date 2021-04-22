#load data
import pandas as pd

df = pd. read_csv('data/test.csv')
print(df.shape, type(df.shape))
print(f'there are {df.shape[0]} rows and {df.shape[1]}columns')
print(df.columns)
print(df.dtypes)
print(df.describe())
print(df.head())

#:understand data
