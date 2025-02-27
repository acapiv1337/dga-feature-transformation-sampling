from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, NearMiss
from imblearn.over_sampling import SMOTE, ADASYN  
from RBU import rbu
import pandas as pd
import numpy as np

def tomek_links(x, y, save_csv=False):
    tomeklink = TomekLinks()
    x_tomek, y_tomek = tomeklink.fit_resample(x, y)
    df_tomek = pd.DataFrame(x_tomek, columns=x.columns)  # Resampled features
    df_tomek['act'] = y_tomek  # Add the resampled target
    if save_csv:
        df_tomek.to_csv('data/balanced/tomek_links.csv', index=True)
    return x_tomek, y_tomek, df_tomek

def smote(x, y, save_csv=False):
    smote = SMOTE()
    x_smote, y_smote = smote.fit_resample(x, y)
    df_smote = pd.DataFrame(x_smote, columns=x.columns)  # Resampled features
    df_smote['act'] = y_smote  # Add the resampled target
    if save_csv:
        df_smote.to_csv('data/balanced/smote.csv', index=True)
    return x_smote, y_smote, df_smote

def smote_enn(x, y, save_csv=False):
    smote = SMOTE()
    x_smote, y_smote = smote.fit_resample(x, y)
    enn = EditedNearestNeighbours()
    x_enn, y_enn = enn.fit_resample(x_smote, y_smote)
    df_enn = pd.DataFrame(x_enn, columns=x.columns)  # Resampled features
    df_enn['act'] = y_enn  # Add the resampled target
    if save_csv:
        df_enn.to_csv('data/balanced/smote_enn.csv', index=True)
    return x_enn, y_enn, df_enn

def smote_adasyn(x, y, save_csv=False):
    smote = SMOTE()
    x_smote, y_smote = smote.fit_resample(x, y)
    adasyn = ADASYN()
    x_adasyn, y_adasyn = adasyn.fit_resample(x_smote, y_smote)
    df_adasyn = pd.DataFrame(x_adasyn, columns=x.columns)  # Resampled features
    df_adasyn['act'] = y_adasyn  # Add the resampled target
    if save_csv:
        df_adasyn.to_csv('data/balanced/smote_adasyn.csv', index=True)
    return x_adasyn, y_adasyn, df_adasyn

def rbu_sampling(x, y, save_csv=False):
    model = rbu.RBU()
    x_rbu, y_rbu = model.fit_sample(np.array(x), y)
    df_rbu = pd.DataFrame(x_rbu, columns=x.columns)
    df_rbu['act'] = y_rbu
    df = pd.DataFrame(x, columns=x.columns)
    df['act'] = y
    for i in df_rbu['act'].unique():
        df = df[df['act'] != i].reset_index(drop=True)
        df = pd.concat([df,df_rbu[df_rbu['act'] == i]]).reset_index(drop=True)
    if save_csv:
        df.to_csv('data/balanced/rbu.csv', index=True)
    return x_rbu, y_rbu, df_rbu

def enn(x, y, save_csv=False):
    enn = EditedNearestNeighbours()
    x_enn, y_enn = enn.fit_resample(x, y)
    df_enn = pd.DataFrame(x_enn, columns=x.columns)  # Resampled features
    df_enn['act'] = y_enn  # Add the resampled target
    if save_csv:
        df_enn.to_csv('data/balanced/enn.csv', index=True)
    return x_enn, y_enn, df_enn

def nearmiss(x, y, save_csv=False):
    nearmiss = NearMiss()
    x_nearmiss, y_nearmiss = nearmiss.fit_resample(x, y)
    df_nearmiss = pd.DataFrame(x_nearmiss, columns=x.columns)  # Resampled features
    df_nearmiss['act'] = y_nearmiss  # Add the resampled target
    if save_csv:
        df_nearmiss.to_csv('data/balanced/nearmiss.csv', index=True)
    return x_nearmiss, y_nearmiss, df_nearmiss

def adasyn(x, y , save_csv='False'):
    adasyn = ADASYN()
    x_adasyn, y_adasyn = adasyn.fit_resample(x, y)
    df_adasyn = pd.DataFrame(x_adasyn, columns=x.columns)  # Resampled features
    df_adasyn['act'] = y_adasyn
    if save_csv:
        df_adasyn.to_csv('data/balanced/adasynn.csv', index=True)
    return x_adasyn, y_adasyn, df_adasyn

def smote_nearmiss(x, y, save_csv=False):
    smote = SMOTE()
    x_smote, y_smote = smote.fit_resample(x, y)
    nearmiss = NearMiss()
    x_nearmiss, y_nearmiss = nearmiss.fit_resample(x_smote, y_smote)
    df_nearmiss = pd.DataFrame(x_nearmiss, columns=x.columns)  # Resampled features
    df_nearmiss['act'] = y_nearmiss  # Add the resampled target
    if save_csv:
        df_nearmiss.to_csv('data/balanced/smote_nearmiss.csv', index=True)
    return x_nearmiss, y_nearmiss, df_nearmiss

def adasyn_enn(x, y, save_csv=False):
    adasyn = ADASYN()
    x_adasyn, y_adasyn = adasyn.fit_resample(x, y)
    enn = EditedNearestNeighbours()
    x_enn, y_enn = enn.fit_resample(x_adasyn, y_adasyn)
    df_enn = pd.DataFrame(x_enn, columns=x.columns)  # Resampled features
    df_enn['act'] = y_enn  # Add the resampled target
    if save_csv:
        df_enn.to_csv('data/balanced/adasyn_enn.csv', index=True)
    return x_enn, y_enn, df_enn

def adasyn_nearmiss(x, y, save_csv=False):
    adasyn = ADASYN()
    x_adasyn, y_adasyn = adasyn.fit_resample(x, y)
    nearmiss = NearMiss()
    x_nearmiss, y_nearmiss = nearmiss.fit_resample(x_adasyn, y_adasyn)
    df_nearmiss = pd.DataFrame(x_nearmiss, columns=x.columns)  # Resampled features
    df_nearmiss['act'] = y_nearmiss  # Add the resampled target
    if save_csv:
        df_nearmiss.to_csv('data/balanced/adasyn_nearmiss.csv', index=True)
    return x_nearmiss, y_nearmiss, df_nearmiss