import pandas as pd
import os
import pkg_resources

# Global variables to store DataFrames, initialized as None
foods_df = None
nutrients_df = None

def ensure_data_loaded():
    """Load the CSV files into DataFrames if not already loaded."""
    global foods_df, nutrients_df
    if foods_df is None:
        data_path = pkg_resources.resource_filename('pyfooda', 'data/fooddata.csv')
        foods_df = pd.read_csv(data_path)
    if nutrients_df is None:
        data_path = pkg_resources.resource_filename('pyfooda', 'data/nutrients.csv')
        nutrients_df = pd.read_csv(data_path)

def get_category(foodName):
    """Return the category of the specified food item."""
    ensure_data_loaded()
    row = foods_df[foods_df['foodName'].str.lower() == foodName.lower()]
    if row.empty:
        return 'Other'
    return row['category'].iloc[0]

def get_nutrients(foodName):
    """Return a dictionary of nutrient values for the specified food item."""
    ensure_data_loaded()
    row = foods_df[foods_df['foodName'].str.lower() == foodName.lower()]
    if row.empty:
        return None
    # Get list of nutrient names from nutrients.csv
    nutrient_columns = nutrients_df['nutrientName'].tolist()
    # Extract nutrient values and convert to dictionary
    nutrients = row[nutrient_columns].iloc[0].to_dict()
    return nutrients

def get_portion_gram_weight(foodName):
    """Return the portion gram weight of the specified food item as a float."""
    ensure_data_loaded()
    row = foods_df[foods_df['foodName'].str.lower() == foodName.lower()]
    if row.empty:
        return None
    return float(row['portion_gram_weight'].iloc[0])

def get_portion_unit_name(foodName):
    """Return the portion unit name of the specified food item."""
    ensure_data_loaded()
    row = foods_df[foods_df['foodName'].str.lower() == foodName.lower()]
    if row.empty:
        return None
    return row['portion_unit_name'].iloc[0]

def find_closest_matches(partialName):
    """Return up to 10 food names that contain the partial name."""
    ensure_data_loaded()
    # Case-insensitive partial match; na=False handles potential NaN values
    mask = foods_df['foodName'].str.lower().str.contains(partialName.lower(), na=False)
    matches = foods_df[mask]['foodName'].tolist()
    return matches[:10]

def get_fooddata_df():
    """Return the fooddata DataFrame."""
    ensure_data_loaded()
    return foods_df

def get_drv_df():
    """Return the DRV DataFrame."""
    ensure_data_loaded()
    return nutrients_df