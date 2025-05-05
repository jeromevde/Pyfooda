#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer  # For smarter text embeddings
from hdbscan import HDBSCAN
from scipy.spatial.distance import cdist
from scipy.sparse import hstack
from tqdm import tqdm  # For progress bar

# fix hack SSL error huggingface
import requests
from huggingface_hub import configure_http_backend
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session
configure_http_backend(backend_factory=backend_factory)

# Load data (first 1000 rows)
df = pd.read_csv('data/fooddata.csv').head(10000)

# Preserve the original DataFrame with untransformed nutrient data
df_original = df.copy()

# Fill missing foodName values
df['foodName'] = df['foodName'].fillna('')

# Define nutrient columns
nutrient_cols = [
    'Energy', 'Carbohydrate', 'Sugars, Total', 'Fiber', 'Total Sugars',
    'Fiber (AOAC 2011.25)', 'Total fat', 'Cholesterol', 'Fatty acids, total saturated',
    'DHA', 'EPA', 'ALA', 'Protein', 'Vitamin A, RAE', 'Vitamin E',
    'Vitamin D (D2 + D3)', 'Vitamin C', 'Thiamin', 'Riboflavin', 'Niacin',
    'Pantothenic acid', 'Vitamin B-6', 'Folate, total', 'Vitamin B-12', 'Choline',
    'Vitamin K2 MK-4', 'Vitamin K1', 'Vitamin A', 'Calcium', 'Iron', 'Magnesium',
    'Phosphorus', 'Potassium', 'Sodium', 'Zinc', 'Copper', 'Iodine', 'Manganese',
    'Molybdenum', 'Selenium'
]

print("Imputing missing values")
imputer = SimpleImputer(strategy='constant', fill_value=0)
df[nutrient_cols] = imputer.fit_transform(df[nutrient_cols])

print("Standardizing nutrient values")
scaler = StandardScaler()
df[nutrient_cols] = scaler.fit_transform(df[nutrient_cols])

print("Generating text embeddings with SentenceTransformer")
# Use a pre-trained sentence transformer model for smarter embeddings
text_model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = text_model.encode(df['foodName'].tolist(), show_progress_bar=True)

print("Applying one-hot encoding for food_category")
ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
category_matrix = ohe.fit_transform(df[['food_category']])

print("Combining features (sparse for category, dense for text embeddings and nutrients)")
# Convert text embeddings to a dense matrix
text_matrix = np.array(text_embeddings)
# Combine with nutrient data (already dense) and category matrix (sparse)
feature_matrix = hstack([category_matrix, text_matrix, df[nutrient_cols]])

print("Generating dense array for clustering")
feature_matrix_dense = feature_matrix.toarray()

print("Applying HDBSCAN")
clusterer = HDBSCAN(min_cluster_size=5, min_samples=1, cluster_selection_method='eom')
labels = clusterer.fit_predict(feature_matrix_dense)

# Assign cluster labels to DataFrame
df['cluster'] = labels
df_original['cluster'] = labels  # Add cluster labels to the original DataFrame

# Function to find the medoid of a cluster
def find_medoid(cluster_indices, feature_matrix):
    cluster_data = feature_matrix[cluster_indices]
    distances = cdist(cluster_data, cluster_data, metric='euclidean')
    medoid_idx = np.argmin(distances.sum(axis=1))
    return cluster_indices[medoid_idx]

# Process clusters to assign representative names
print("Processing clusters")
clusters = df['cluster'].unique()
cluster_to_rep_name = {}  # Map cluster label to representative name
for cluster in tqdm(clusters, desc="Processing clusters"):
    if cluster == -1:  # Noise points
        cluster_to_rep_name[cluster] = 'Noise'
        continue
    cluster_df = df[df['cluster'] == cluster]
    cluster_indices = cluster_df.index.to_numpy()
    medoid_idx = find_medoid(cluster_indices, feature_matrix_dense)
    representative_name = df.loc[medoid_idx, 'foodName']
    cluster_to_rep_name[cluster] = representative_name

# Assign representative names to both DataFrames
df['representative_name'] = df['cluster'].map(cluster_to_rep_name)
df_original['representative_name'] = df_original['cluster'].map(cluster_to_rep_name)

# Set representative name as index and sort
df_original.set_index('representative_name', inplace=True)
df_original.sort_index(inplace=True)

# Write the resulting DataFrame to a CSV file (with untransformed nutrient data)
output_file = 'data/fooddata_clustered.csv'
df_original.to_csv(output_file)
print(f"Results written to {output_file}")

#%%