import pandas as pd


def epitope_dataset():
    file_path_train = './datasets/bepi3_epitopes.csv'
    file_path_eval = './datasets/bepi3_epitopes_eval.csv'
    df_train = pd.read_csv(file_path_train)
    df_eval = pd.read_csv(file_path_eval)

    return df_train.loc[df_train['PEPTIDE'].str.len() <= 30],  df_eval.loc[df_eval['PEPTIDE'].str.len() <= 30]
