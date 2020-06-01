import numpy as np
import pandas as pd
from tqdm import tqdm
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
import os

df_train = pd.read_csv('train_kaggle.csv')
df_test = pd.read_csv('sample_solution.csv')
y = df_train['label']

def load_dataframe(id):
    train_data = np.load("train/train/{}.npy".format(id))
    return pd.DataFrame(data=train_data)

def load_dataframe_test(id):
    train_data = np.load("test/test/{}.npy".format(id))
    return pd.DataFrame(data=train_data)

def get_imputed_dataframes(col, average):
    dataframes = []

    for id in tqdm(df_train['Id']):
        df = load_dataframe(id)[[col]]
        dfi = df.fillna(average[col])
        dfi['Id'] = id

        dataframes.append(dfi)
    return dataframes

def get_imputed_test_dataframes(col, average):
    dataframestest = []
    
    for id in tqdm(df_test['Id']):
        df = load_dataframe_test(id)[[col]]
        dfi = df.fillna(average[col])
        dfi['Id'] = id
        
        dataframestest.append(dfi)
    return dataframestest
    
NUM_OF_COL = 40

# Calculate average for each feature column and store it as npy
average = np.empty(NUM_OF_COL)
accumulated_count = [0] * df_train.shape[0]
for id in tqdm(df_train['Id']):
    df = load_dataframe(id)
    for i in range(NUM_OF_COL):
        values = df.values[:,i]
        is_not_nan = ~np.isnan(values)
        values = values[is_not_nan]
        if len(values) == 0:
            continue
        count = len(values)
        average[i] = (average[i] * accumulated_count[i] + np.sum(values)) / (accumulated_count[i] + count)
        accumulated_count[i] += count
np.save("average", average)

# Create all required folders
required_folders = ["features", "filtered_features", "test_features", "filtered_test_features"]
for folder in required_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
    
# Generate and save extracted features for each feature column for train data
for i in range(NUM_OF_COL):
    print("feature", str(i))
    df_train_all = pd.concat(get_imputed_dataframes(i, average), ignore_index=True)
    extracted_features = extract_features(df_train_all, column_id="Id", show_warnings=False)
    extracted_features.to_parquet("features/feature" + str(i) + ".gzip", compression="gzip")

# Generate and save selected features for train data
for i in range(NUM_OF_COL):
    print("filtered feature", str(i))
    extracted_features = pd.read_parquet("features/feature" + str(i) + ".gzip")
    impute(extracted_features)
    features_filtered = select_features(extracted_features, y)
    features_filtered.to_parquet("filtered_features/filtered_feature" + str(i) + ".gzip", compression="gzip")

# Generate and save extracted features for each feature column for test data
for i in range(NUM_OF_COL):
    print("test feature", str(i))
    df_test_all = pd.concat(get_imputed_dataframes(i, average), ignore_index=True)
    extracted_features_test = extract_features(df_test_all, column_id="Id", show_warnings=False)
    extracted_features_test.to_parquet("test_features/test_feature" + str(i) + ".gzip", compression="gzip")

# Generate and save selected extracted features for each feature column for test data
for i in range(NUM_OF_COL):
    print("filtered test feature", str(i))
    extracted_features_test = pd.read_parquet("test_features/test_feature" + str(i) + ".gzip")
    features_filtered = pd.read_parquet("filtered_features/filtered_feature" + str(i) + ".gzip")
    test_features_filtered = extracted_features_test[features_filtered.columns]
    test_features_filtered.to_parquet("filtered_test_features/filtered_test_feature" + str(i) + ".gzip", compression="gzip")
