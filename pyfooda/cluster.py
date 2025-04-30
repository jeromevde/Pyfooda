#%%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from hdbscan import HDBSCAN
from scipy.spatial.distance import cdist
import re
from tqdm import tqdm  # Import tqdm for progress bar

df = pd.read_csv('data/fooddata.csv').head(10000)

df['original_foodName'] = df['foodName']

df['foodName'] = df['foodName'].fillna('unknown')

df['foodName'] = df['foodName'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))

nutrient_cols = list(df.columns[6:])[:-1]

print("imputting missing values")
imputer = SimpleImputer(strategy='constant', fill_value=0)
df[nutrient_cols] = imputer.fit_transform(df[nutrient_cols])

print("Standardize nutrient values")
scaler = StandardScaler()
df[nutrient_cols] = scaler.fit_transform(df[nutrient_cols])

print("tfidf")
tfidf = TfidfVectorizer(max_features=1000, min_df=5)
tfidf_matrix = tfidf.fit_transform(df['foodName'])

print("onehot")
ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
category_matrix = ohe.fit_transform(df[['food_category']])

print("Combine features (keep sparse for TF-IDF and category, dense for nutrients)")
from scipy.sparse import hstack
feature_matrix = hstack([tfidf_matrix, category_matrix, df[nutrient_cols]])

print("array generation")
feature_matrix_dense = feature_matrix.toarray()

print("Apply HDBSCAN")
clusterer = HDBSCAN(min_cluster_size=5, min_samples=1, cluster_selection_method='eom')
labels = clusterer.fit_predict(feature_matrix_dense)

df['cluster'] = labels

# Function to find the medoid of a cluster
def find_medoid(cluster_indices, feature_matrix):
    cluster_data = feature_matrix[cluster_indices]
    distances = cdist(cluster_data, cluster_data, metric='euclidean')
    medoid_idx = np.argmin(distances.sum(axis=1))
    return cluster_indices[medoid_idx]

#%%
print("process_clusters")
clusters = df['cluster'].unique()
for cluster in tqdm(clusters, desc="Processing clusters"):
    if cluster == -1:  # Noise points
        continue
    cluster_df = df[df['cluster'] == cluster]
    cluster_indices = cluster_df.index.to_numpy()
    medoid_idx = find_medoid(cluster_indices, feature_matrix_dense)
    representative_name = df.loc[medoid_idx, 'foodName']
    average_nutrients = cluster_df[nutrient_cols].mean()
    food_names = cluster_df['original_foodName'].tolist()

    
    print(f"Cluster {cluster}:")
    print(food_names)
    print(f"  Representative Name: {representative_name}")
    print(f"  Average Nutrients:")
    print(average_nutrients.to_string())
    print("\n")
# %%
