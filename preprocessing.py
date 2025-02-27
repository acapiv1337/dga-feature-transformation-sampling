import numpy as np
from scipy.stats import boxcox, yeojohnson

def add_epsilon(df, save_csv='False'):
    for col in df.columns:
        if col != 'act':
            df[col] = df[col] + np.finfo(float).eps
    if save_csv:
        df.to_csv('data/data transform/epsilon.csv', index=True)
    return df

def transform_log(df, save_csv='False'):
    for col in df.columns:
        if col != 'act':
            df[col] = np.log(df[col])
    if save_csv:
        df.to_csv('data/data transform/log.csv', index=True)
    return df

def transform_log1p(df, save_csv='False'):
    for col in df.columns:
        if col != 'act':
            df[col] = np.log1p(df[col])
    if save_csv:
        df.to_csv('data/data transform/log1p.csv', index=True)
    return df.copy()

def transform_sqrt(df, save_csv='False'):
    for col in df.columns:
        if col != 'act':
            df[col] = np.sqrt(df[col])
    if save_csv:
        df.to_csv('data/data transform/sqrt.csv', index=True)
    return df

def transform_boxcox(df, save_csv='False'):
    for col in df.columns:
        if col != 'act':
            df[col] = boxcox(df[col])[0]
    if save_csv:
        df.to_csv('data/data transform/boxcox.csv', index=True)
    return df

def transform_yeojohnson(df, save_csv='False'):
    for col in df.columns:
        if col != 'act':
            df[col] = yeojohnson(df[col])[0]
    if save_csv:
        df.to_csv('data/data transform/yeojohnson.csv', index=True)
    return df

def change_column_name(df):
    df['act'] = df['act'].astype(str)  # Convert the column to string type
    df['act'].replace({'1': 'PD', '2': 'D1', '3': 'D2', '4': 'T1', '5': 'T2', '6': 'T3'}, inplace=True)
    return df

def softmax(predict_softmax):
    predict = predict_softmax.argmax(axis=1)
    return predict