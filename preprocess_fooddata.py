#%%
import pandas as pd
from nutrients_drv import get_nutrients_with_drv_df
import os 

fooddata_folder = "/Users/jf41043/Downloads/FoodData_Central_csv_2024-10-31"

#%% --- Load food and category data ---
food_df = pd.read_csv(f'{fooddata_folder}/food.csv', usecols=['fdc_id', 'data_type', 'food_category_id', 'description'])
food_df = food_df.rename(columns={"description": "foodName"})
food_category_df = pd.read_csv(f'{fooddata_folder}/food_category.csv', usecols=['id', 'description']).rename(columns={"id": "category_id", "description": "category_description"})
food_df['category_id_int'] = pd.to_numeric(food_df['food_category_id'], errors='coerce')
food_df = pd.merge(food_df, food_category_df[['category_id', 'category_description']], left_on='category_id_int', right_on='category_id', how='left')
food_df['category_description'] = food_df['category_description'].fillna(food_df['food_category_id'])
food_df = food_df.drop(columns=['category_id_int', 'category_id'], errors='ignore')
foundation_foods = food_df

#%% --- Load and merge nutrient data ---
food_nutrient_df = pd.read_csv(f'{fooddata_folder}/food_nutrient.csv', usecols=['id', 'fdc_id', 'nutrient_id', 'amount'])
nutrient_df = get_nutrients_with_drv_df()
nutrient_df["nutrient_order"] = nutrient_df.index

food_nutrients = pd.merge(foundation_foods, food_nutrient_df, on='fdc_id', how='left')
food_nutrients = pd.merge(food_nutrients, nutrient_df, on='nutrient_id', how='left')

#%% --- Convert energy values from kJ to kcal where needed ---
energy_mask = food_nutrients['category'] == 'Energy'
food_nutrients.loc[energy_mask & (food_nutrients['unit_name'] == 'kJ'), 'amount'] = food_nutrients.loc[energy_mask & (food_nutrients['unit_name'] == 'kJ'), 'amount'] / 4.184
food_nutrients.loc[energy_mask & (food_nutrients['unit_name'] == 'kJ'), 'unit_name'] = 'KCAL'

#%% --- Merge with portion data ---
food_portion_df = pd.read_csv(f'{fooddata_folder}/food_portion.csv', usecols=['fdc_id', 'amount', 'gram_weight', 'measure_unit_id'])
food_portion_df = food_portion_df.rename(columns={"amount": "portion_amount", "gram_weight": "portion_gram_weight"})
measure_unit_df = pd.read_csv(f'{fooddata_folder}/measure_unit.csv', usecols=['id', 'name']).rename(columns={"id": "measure_unit_id", "name": "portion_unit_name"})
food_portion_df = pd.merge(food_portion_df, measure_unit_df, on='measure_unit_id', how='left')
food_portion_df["portion_gram_weight"] = food_portion_df["portion_gram_weight"] / food_portion_df["portion_amount"]
food_portion_df = food_portion_df[["fdc_id", "portion_gram_weight", "portion_unit_name"]]
food_nutrients = food_nutrients.merge(food_portion_df, on="fdc_id", how="left")

df = food_nutrients

#%% --- Clean and process data ---
df['name'] = df['name'].str.replace(r'\s*\(.*\)', '', regex=True)
df['name'] = df['name'].str.split(',').str[0].str.strip()
df = df.rename(columns={"name": "nutrientName"})
df = df.groupby(['foodName', 'nutrientName'], sort=False).agg({
    'data_type': 'first',
    'amount': 'mean',
    'unit_name': 'first',
    'category': 'first',
    'drv': 'max',
    'category_description': 'first',
    'nutrient_order': 'first',
    'portion_unit_name': 'first',
    'portion_gram_weight': 'first',
}).reset_index()
df['foodName'] = df['foodName'].str.strip()
df['unit_name'] = df['unit_name'].str.lower()
df = df.sort_values(by=['foodName', 'nutrient_order'])
df['portion_gram_weight'] = df['portion_gram_weight']
df = df.drop_duplicates(subset=['foodName', 'nutrientName'])

df = df[df['drv'].notna()]

#%% --- Filter food types ---
df = df[
    (df["data_type"]=="foundation_food") |  
    (df["data_type"]=="branded_food") | 
    (df["data_type"]=="sr_legacy_food") |
    (df["data_type"]=="survey_fndds_food") 
    ]

#%% --- Filter foods by minimum nutrients ---
min_nutrients = 11
nutrient_counts = df[df['amount'] != ''].groupby('foodName').size()
selected_foods = nutrient_counts[nutrient_counts >= min_nutrients].index
df = df[df['foodName'].isin(selected_foods)]
print(f"Selected {len(df['foodName'].unique())} foods with at least {min_nutrients} non-missing nutrients.")

#%% --- Create and save foods-by-nutrient pivot CSV ---
pivot_df = df.pivot_table(index=['foodName', 'portion_unit_name', 'portion_gram_weight'], columns='nutrientName', values='amount', aggfunc='first').reset_index()
if not os.path.exists("data"):
    os.makedirs("data")
pivot_df.to_csv('data/foods.csv', index=False)

#%% --- Create and save nutrient details CSV based on nutrient_df ---
nutrient_details_df = nutrient_df.rename(columns={"name": "nutrientName"})
nutrient_details_cols = ['nutrientName', 'category', 'unit_name', 'drv', 'nutrient_id', 'nutrient_order']
nutrient_details = nutrient_details_df[nutrient_details_cols].drop_duplicates(subset=['nutrientName']).set_index('nutrientName').sort_index()
nutrient_details.to_csv('data/nutrients.csv')

print("CSV files 'foods_by_nutrient.csv' and 'nutrient_details.csv' created")

# %%
