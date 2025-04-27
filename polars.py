import polars as pl
from nutrients_drv import get_nutrients_with_drv_df  # Assuming this returns a Pandas DataFrame
import os

fooddata_folder = "/Users/jerome/Downloads/FoodData_Central_csv_2024-10-31"

# --- FOOD DATA ---
food_df = pl.read_csv(
    f"{fooddata_folder}/food.csv",
    columns=["fdc_id", "data_type", "food_category_id", "description"]
).rename({"description": "foodName"})

# --- CATEGORY DATA ---
food_category_df = pl.read_csv(
    f"{fooddata_folder}/food_category.csv",
    columns=["id", "description"]
).rename({"id": "category_id", "description": "food_category"})

wweia_categories = pl.read_csv(
    f"{fooddata_folder}/wweia_food_category.csv"
).rename({
    "wweia_food_category": "category_id",
    "wweia_food_category_description": "food_category"
})

food_category_df = pl.concat([wweia_categories, food_category_df])

food_df = food_df.with_columns(
    pl.col("food_category_id").cast(pl.Float64, strict=False).alias("category_id_int")
).join(
    food_category_df.select(["category_id", "food_category"]),
    left_on="category_id_int",
    right_on="category_id",
    how="left"
).with_columns(
    pl.col("food_category").fill_null(pl.col("food_category_id"))
).drop(["category_id_int", "category_id"])

# --- NUTRIENT DATA ---
food_nutrient_df = pl.read_csv(
    f"{fooddata_folder}/food_nutrient.csv",
    columns=["id", "fdc_id", "nutrient_id", "amount"]
)

# Convert Pandas DataFrame from get_nutrients_with_drv_df() to Polars
nutrient_df = pl.from_pandas(get_nutrients_with_drv_df())
nutrient_df = nutrient_df.with_row_index("nutrient_order")

food_nutrients = food_df.join(
    food_nutrient_df, on="fdc_id", how="left"
).join(
    nutrient_df, on="nutrient_id", how="left"
)

# --- Convert energy values from kJ to kcal where needed ---
food_nutrients = food_nutrients.with_columns(
    pl.when(
        (pl.col("nutrient_category") == "Energy") & (pl.col("unit_name") == "kJ")
    ).then(
        pl.col("amount") / 4.184
    ).otherwise(
        pl.col("amount")
    ).alias("amount"),
    pl.when(
        (pl.col("nutrient_category") == "Energy") & (pl.col("unit_name") == "kJ")
    ).then(
        pl.lit("KCAL")
    ).otherwise(
        pl.col("unit_name")
    ).alias("unit_name")
)

# --- PORTION DATA ---
food_portion_cols = ["fdc_id", "amount", "gram_weight", "measure_unit_id"]
food_portion_df = pl.read_csv(
    f"{fooddata_folder}/food_portion.csv",
    columns=food_portion_cols
).rename({
    "amount": "portion_amount",
    "gram_weight": "portion_gram_weight"
})

measure_unit_cols = ["id", "name"]
measure_unit_df = pl.read_csv(
    f"{fooddata_folder}/measure_unit.csv",
    columns=measure_unit_cols
).rename({"id": "measure_unit_id", "name": "portion_unit_name"})

food_portion_df = food_portion_df.join(
    measure_unit_df, on="measure_unit_id", how="left"
).with_columns(
    (pl.col("portion_gram_weight") / pl.col("portion_amount")).alias("portion_gram_weight")
).select(["fdc_id", "portion_gram_weight", "portion_unit_name"])

food_nutrients = food_nutrients.join(
    food_portion_df, on="fdc_id", how="left"
)

# --- Checkpoint of the joins ---
df = food_nutrients

# --- Remove non-drv nutrients ---
df = df.filter(pl.col("drv").is_not_null())

# --- GROUP (Pivot) ---
index_cols = ["foodName", "data_type", "food_category", "portion_unit_name", "portion_gram_weight"]
columns_col = "nutrientName"

# Fill nulls with empty string for pivot compatibility
df = df.with_columns(
    [pl.col(col).fill_null("") for col in index_cols + [columns_col]]
)

# Pivot in Polars using group_by and aggregation
pivot_df = df.group_by(index_cols).agg(
    pl.col("amount").first().alias(col) for col in nutrient_df["nutrientName"].unique()
).fill_null("")

# Calculate number of non-null nutrients
nutrient_cols = [col for col in pivot_df.columns if col not in index_cols]
number_nutrients = pivot_df.select(
    pl.sum_horizontal(
        [(pl.col(col) != "") & pl.col(col).is_not_null() for col in nutrient_cols]
    ).alias("number_of_nutrients")
)
pivot_df = pivot_df.with_columns(number_nutrients["number_of_nutrients"]).select(
    index_cols + ["number_of_nutrients"] + list(nutrient_df["nutrientName"].unique())
)

# --- SAVE NUTRIENT DETAILS ---
nutrient_details_cols = ["nutrientName", "nutrient_category", "unit_name", "drv", "nutrient_id"]
nutrient_details = nutrient_df.select(nutrient_details_cols).unique(subset=["nutrientName"]).sort("nutrientName")

# --- FILTER FOODS ---
pivot_df = pivot_df.filter(
    (pl.col("data_type") == "foundation_food") |
    (pl.col("data_type") == "survey_fndds_food") |
    (pl.col("data_type") == "sr_legacy_food") |
    ((pl.col("data_type") == "branded_food") & (pl.col("foodName").str.len_chars() < 45))
)

# --- SAVE ---
if not os.path.exists("data"):
    os.makedirs("data")

pivot_df.write_csv("data/foods.csv")
pivot_df.write_excel("data/foods.xlsx")

nutrient_df.write_csv("data/nutrients.csv")
nutrient_df.write_excel("data/nutrients.xlsx")