#%%
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from upsetplot import UpSet, from_indicators
import pandas as pd
import matplotlib.pyplot as plt

def load_and_merge_data(fooddata_folder):
    """Load and merge food, nutrient, and portion data from CSV files with specific columns."""
    # Load only necessary columns
    food_df = pd.read_csv(f'{fooddata_folder}/food.csv', 
                         usecols=['fdc_id', 'data_type', 'food_category_id', 'description'])
    food_category_df = pd.read_csv(f'{fooddata_folder}/food_category.csv', 
                                  usecols=['id', 'description']).rename(
        columns={"id": "category_id", "description": "category_description"}
    )
    food_df['category_id_int'] = pd.to_numeric(food_df['food_category_id'], errors='coerce')
    food_df = pd.merge(food_df, 
                       food_category_df[['category_id', 'category_description']],
                       left_on='category_id_int', 
                       right_on='category_id', 
                       how='left')
    food_df['category_description'] = food_df['category_description'].fillna(food_df['food_category_id'])
    food_df = food_df.drop(columns=['category_id_int', 'category_id'], errors='ignore')
    foundation_foods = food_df

    # Merge with nutrient data
    food_nutrient_df = pd.read_csv(f'{fooddata_folder}/food_nutrient.csv', 
                                  usecols=['id', 'fdc_id', 'nutrient_id', 'amount'])
    nutrient_df = pd.read_csv('nutrients_with_categories_and_drv.csv', 
                             usecols=['id', 'name', 'unit_name', 'category', 'drv']).rename(
        columns={"id": "nutrient_id"}
    )
    nutrient_df["nutrient_order"] = nutrient_df.index
    foundation_nutrients = pd.merge(foundation_foods, food_nutrient_df, on='fdc_id', how='left')
    foundation_nutrients = pd.merge(foundation_nutrients, nutrient_df, on='nutrient_id', how='left')

    # Merge with portion data
    food_portion_df = pd.read_csv(f'{fooddata_folder}/food_portion.csv', 
                                 usecols=['fdc_id', 'amount', 'gram_weight', 'measure_unit_id'])
    food_portion_df = food_portion_df.rename(
        columns={"amount": "portion_amount", "gram_weight": "portion_gram_weight"}
    )
    measure_unit_df = pd.read_csv(f'{fooddata_folder}/measure_unit.csv', 
                                 usecols=['id', 'name']).rename(
        columns={"id": "measure_unit_id", "name": "portion_unit_name"}
    )
    food_portion_df = pd.merge(food_portion_df, measure_unit_df, on='measure_unit_id', how='left')
    food_portion_df["portion_gram_weight"] = food_portion_df["portion_gram_weight"] / food_portion_df["portion_amount"]
    food_portion_df = food_portion_df[["fdc_id", "portion_gram_weight", "portion_unit_name"]]
    foundation_nutrients = foundation_nutrients.merge(food_portion_df, on="fdc_id", how="left")

    return foundation_nutrients

def clean_and_process_data(df):
    """Clean and aggregate the merged data."""
    df['name'] = df['name'].str.replace(r'\s*\(.*\)', '', regex=True)
    df['name'] = df['name'].str.split(',').str[0].str.strip()
    df = df.groupby(['description', 'name'], sort=False).agg({
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
    df['description'] = df['description'].str.strip()
    df['unit_name'] = df['unit_name'].str.lower()
    df = df.sort_values(by=['description', 'nutrient_order'])
    df['amount'] = df['amount'].fillna('N/A')
    df['portion_gram_weight'] = df['portion_gram_weight'].fillna('N/A')
    df = df.drop_duplicates(subset=['description', 'name'])
    return df

def filter_foods(df, min_nutrients=None):
    """Filter foods to include only those with at least min_nutrients non-missing nutrients."""
    if min_nutrients is None:
        return df
    nutrient_counts = df[df['amount'] != 'N/A'].groupby('description').size()
    selected_foods = nutrient_counts[nutrient_counts >= min_nutrients].index
    return df[df['description'].isin(selected_foods)]

def create_csv(df):
    """Create a CSV file from the processed DataFrame with a #nutrients column."""
    # Create nutrient column identifier
    df['nutrient_col'] = df['name'] + '|' + df['category'] + '|' + df['drv'].astype(str) + '|' + df['unit_name']
    
    # Pivot to get nutrients as columns
    pivot_df = df.pivot_table(index='description', 
                             columns='nutrient_col', 
                             values='amount', 
                             aggfunc='first').reset_index()
    
    # Extract food-level info
    food_info = df.groupby('description').agg({
        'category_description': 'first',
        'data_type': "first",
        'portion_unit_name': 'first',
        'portion_gram_weight': 'first'
    }).reset_index()
    
    # Calculate number of non-'N/A' nutrients
    nutrient_counts = df[df['amount'] != 'N/A'].groupby('description').size().reset_index(name='#nutrients')
    
    # Merge into final CSV DataFrame
    csv_df = pd.merge(pivot_df, food_info, on='description')
    csv_df = pd.merge(csv_df, nutrient_counts, on='description')
    
    # Rename and reorder columns
    csv_df = csv_df.rename(columns={'description': 'food_name', 'category_description': 'category'})
    nutrient_cols = [col for col in csv_df.columns if '|' in col]
    csv_df = csv_df[['food_name', 'data_type', 'category', 'portion_unit_name', 'portion_gram_weight', '#nutrients'] + nutrient_cols]
    
    return csv_df

def create_json_dict(df):
    """Create a dictionary for JSON output from the processed DataFrame."""
    final_dict = {}
    for description, group in df.groupby('description'):
        category_desc = group['category_description'].iloc[0]
        portion_unit_name = group['portion_unit_name'].iloc[0]
        portion_gram_weight = group['portion_gram_weight'].iloc[0]
        group = group.set_index('name')
        nutrients = group[['amount', 'unit_name', 'category', 'drv']].to_dict(orient='index')
        final_dict[description] = {
            'category': category_desc,
            'portion_unit_name': portion_unit_name,
            'portion_gram_weight': portion_gram_weight,
            'nutrients': nutrients
        }
    return final_dict


def plot_distribution(df):
    nutrient_counts = df[df['amount'] != 'N/A'].groupby('description').size()
    plt.figure(figsize=(10, 6))
    min_count = int(nutrient_counts.min())
    max_count = int(nutrient_counts.max())
    bins = np.arange(min_count - 0.5, max_count + 1.5)
    plt.hist(nutrient_counts, bins=bins, edgecolor='black')
    plt.title('Distribution of Number of Nutrients per Food')
    plt.xlabel('Number of Nutrients')
    plt.ylabel('Number of Foods')
    plt.grid(True)
    plt.show()


def plot_upset(df, max_combinations=20):
    """
    Plot an UpSet plot showing the most common nutrient combinations across foods.

    Parameters:
    - df: DataFrame with processed nutrient data (columns: 'description', 'name', 'amount', etc.).
    - max_combinations: Maximum number of nutrient combinations to display (default: 20).
    """
    binary_df = df.pivot_table(index='description',
                               columns='name',
                               values='amount',
                               aggfunc=lambda x: (x != 'N/A').any()).fillna(False)
    upset_data = from_indicators(binary_df.columns, data=binary_df)
    upset = UpSet(upset_data, subset_size='count', show_counts=True)
    upset.plot()
    plt.title('UpSet Plot of Nutrient Combinations Across Foods')
    plt.show()



#%%
fooddata_folder = "/Users/jf41043/Downloads/FoodData_Central_csv_2024-10-31"
data = load_and_merge_data(fooddata_folder)
data = clean_and_process_data(data)
data = data[data['drv'].notna()]

#%% Plot nutrient distribution
df = data

df = df[
        (df["data_type"]=="foundation_food") |  
        (df["data_type"]=="branded_food") | 
        (df["data_type"]=="sr_legacy_food") |
        (df["data_type"]=="survey_fndds_food") 
        ]
plot_distribution(df)
#plot_upset(df)

#%% Filter foods
min_nutrients=1
df = filter_foods(df, min_nutrients)
print(f"Selected {len(df['description'].unique())} foods with at least {min_nutrients} non-missing nutrients.")


#%% Create CSV first

csv_df = create_csv(df)
csv_df.to_excel('../data/fooddata.xlsx', index=False, na_rep='')
csv_df.to_csv('../data/fooddata.csv', index=False, na_rep='')

print("CSV created")


if False:
    json_dict = create_json_dict(df)
    with open('../data/fooddata.json', 'w') as fp:
        json.dump(json_dict, fp)
    print("JSON dictionary created")

