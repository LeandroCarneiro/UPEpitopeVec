import pandas as pd


def epitope_dataset():
    # Specify the path to your CSV file
    file_path = './datasets/bepi3_epitopes_test.csv'
    df = pd.read_csv(file_path)

    # print(df)
    # positive_epitopes = df[df['FLAG'] == '1'].to_numpy()
    # negative_epitopes = df[df['FLAG'] == '0'].to_numpy()
    return df.loc[df['PEPTIDE'].str.len() <= 30]
    # return positive_epitopes, negative_epitopes


# Example usage
# positive_epitopes, negative_epitopes = epitope_classifier()
# result = len(positive_epitopes)
# print(f"positive: {result}")
# result = len(negative_epitopes)
# print(f"negative: {result}")
