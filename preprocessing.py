import numpy as np
from scipy.stats import boxcox, yeojohnson

def add_epsilon(df, save_csv='False', label_column='act', save_path='data/data transform/epsilon.csv'):
    for col in df.columns:
        if col != label_column:
            df[col] = df[col] + np.finfo(float).eps
    if save_csv:
        df.to_csv(save_path, index=True)
    return df

def transform_log(df, save_csv='False', label_column='act', save_path='data/data transform/log.csv'):
    for col in df.columns:
        if col != label_column:
            df[col] = np.log(df[col])
    if save_csv:
        df.to_csv(save_path, index=True)
    return df

def transform_log1p(df, save_csv='False', label_column='act', save_path='data/data transform/log1p.csv'):
    for col in df.columns:
        if col != label_column:
            df[col] = np.log1p(df[col])
    if save_csv:
        df.to_csv(save_path, index=True)
    return df.copy()

def transform_sqrt(df, save_csv='False', label_column='act', save_path='data/data transform/sqrt.csv'):
    for col in df.columns:
        if col != label_column:
            df[col] = np.sqrt(df[col])
    if save_csv:
        df.to_csv(save_path, index=True)
    return df

def transform_boxcox(df, save_csv='False', label_column='act', save_path='data/data transform/boxcox.csv):
    for col in df.columns:
        if col != label_column:
            df[col] = boxcox(df[col])[0]
    if save_csv:
        df.to_csv(save_path, index=True)
    return df

def transform_yeojohnson(df, save_csv='False', label_column='act', save_path='data/data transform/yeojohnson.csv'):
    for col in df.columns:
        if col != label_column:
            df[col] = yeojohnson(df[col])[0]
    if save_csv:
        df.to_csv(save_path, index=True)
    return df

def change_column_name(df):
    df[label_column] = df[label_column].astype(str)  # Convert the column to string type
    df[label_column].replace({'1': 'PD', '2': 'D1', '3': 'D2', '4': 'T1', '5': 'T2', '6': 'T3'}, inplace=True)
    return df

def softmax(predict_softmax):
    predict = predict_softmax.argmax(axis=1)
    return predict
