#%%
import pandas as pd
from nutrients_drv import get_nutrients_with_drv_df
import os 

fooddata_folder = "/Users/jerome/Downloads/FoodData_Central_csv_2024-10-31"

#%% --- FOOD DATA ---
food_df = pd.read_csv(f'{fooddata_folder}/food.csv', usecols=['fdc_id', 'data_type', 'food_category_id', 'description'])
food_df = food_df.rename(columns={"description": "foodName"})
food_df['foodName'] = food_df['foodName'].str.replace('"', '', regex=False).str.strip()

#%% --- CATEGORY DATA ---
food_category_df = pd.read_csv(f'{fooddata_folder}/food_category.csv', usecols=['id', 'description'])
food_category_df = food_category_df.rename(columns={"id": "category_id", "description": "food_category"})
wweia_categories = pd.read_csv(f'{fooddata_folder}/wweia_food_category.csv')
wweia_categories=wweia_categories.rename(columns={"wweia_food_category": "category_id","wweia_food_category_description": "food_category"})
food_category_df = pd.concat([wweia_categories, food_category_df])
food_df['category_id_int'] = pd.to_numeric(food_df['food_category_id'], errors='coerce')
food_df = pd.merge(food_df, food_category_df[['category_id', 'food_category']], left_on='category_id_int', right_on='category_id', how='left')
food_df['food_category'] = food_df['food_category'].fillna(food_df['food_category_id']) # category_id sometimes already contains the food_category
food_df = food_df.drop(columns=['category_id_int', 'category_id'], errors='ignore')


#%% --- NUTRIENT DATA ---
food_nutrient_df = pd.read_csv(f'{fooddata_folder}/food_nutrient.csv', usecols=['id', 'fdc_id', 'nutrient_id', 'amount'])
nutrient_df = get_nutrients_with_drv_df()
nutrient_df["nutrient_order"] = nutrient_df.index
food_nutrients = pd.merge(food_df, food_nutrient_df, on='fdc_id', how='left')
food_nutrients = pd.merge(food_nutrients, nutrient_df, on='nutrient_id', how='left')

#%% --- Convert energy values from kJ to kcal where needed ---
mask = (food_nutrients['nutrient_category'] == 'Energy') & (food_nutrients['unit_name'] == 'kJ')
food_nutrients.loc[mask , 'amount'] = food_nutrients.loc[mask, 'amount'] / 4.184
food_nutrients.loc[mask, 'unit_name'] = 'KCAL'

#%% --- PORTION DATA ---
food_portion_cols = ['fdc_id', 'amount', 'gram_weight', 'measure_unit_id']
food_portion_df = pd.read_csv(f'{fooddata_folder}/food_portion.csv', usecols=food_portion_cols)
food_portion_df = food_portion_df.rename(columns={"amount": "portion_amount", "gram_weight": "portion_gram_weight"})
measure_unit_cols = ['id', 'name']
measure_unit_df = pd.read_csv(f'{fooddata_folder}/measure_unit.csv', usecols=measure_unit_cols)
measure_unit_df = measure_unit_df.rename(columns={"id": "measure_unit_id", "name": "portion_unit_name"})
food_portion_df = pd.merge(food_portion_df, measure_unit_df, on='measure_unit_id', how='left')
food_portion_df["portion_gram_weight"] = food_portion_df["portion_gram_weight"] / food_portion_df["portion_amount"]
food_portion_df = food_portion_df[["fdc_id", "portion_gram_weight", "portion_unit_name"]]
food_nutrients = food_nutrients.merge(food_portion_df, on="fdc_id", how="left")

#%% --- Checkpoint of the joins ---
df = food_nutrients


#%% --- Remove non-drv nutrients ---
df = df[df['drv'].notna()]


#%% --- GROUP ---
index_cols = ['foodName', 'data_type', 'food_category', 'portion_unit_name', 'portion_gram_weight']
columns_col = 'nutrientName'
df[index_cols + [columns_col]] = df[index_cols + [columns_col]].fillna('') # so the pivot doesn't remove rows with na values
pivot_df = df.pivot_table(
    index=index_cols,
    columns=columns_col,
    values='amount',
    aggfunc='first'
).reset_index()
nutrient_cols = [col for col in pivot_df.columns if col not in index_cols]
pivot_df = pivot_df[index_cols +list(nutrient_df["nutrientName"].unique())] # preserve nutrient order
number_nutrients = ((pivot_df[nutrient_cols] != "") & (pivot_df[nutrient_cols].notna())).sum(axis=1)
pivot_df.insert(loc=5, column='number_of_nutrients', value=number_nutrients)


#%% --- save NUTRIENT DETAILS ---
nutrient_details_cols = ['nutrientName', 'nutrient_category', 'unit_name', 'drv', 'nutrient_id']
nutrient_details = nutrient_df[nutrient_details_cols].drop_duplicates(subset=['nutrientName']).set_index('nutrientName').sort_index()


# %% --- FILTER FOODS ---
pivot_df = pivot_df[pivot_df["foodName"].str.len() < 45]


#%% --- SAVE ---
if not os.path.exists("data"):
    os.makedirs("data")

pivot_df.to_csv('data/foods.csv', index=False)
pivot_df.to_excel('data/foods.xlsx', index=False)


nutrient_df.to_csv('data/nutrients.csv')
nutrient_df.to_excel('data/nutrients.xlsx')
# %%
