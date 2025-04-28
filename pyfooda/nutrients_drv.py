import pandas as pd

def get_nutrients_with_drv():
    """
    Return the hardcoded list of nutrients with non-zero DRV values.
    """
    nutrients = [
        {'nutrient_id': 2047, 'name': 'Energy', 'unit_name': 'KCAL', 'category': 'Energy', 'drv': 2500.0},
        {'nutrient_id': 2048, 'name': 'Energy', 'unit_name': 'KCAL', 'category': 'Energy', 'drv': 2500.0},
        {'nutrient_id': 1008, 'name': 'Energy', 'unit_name': 'KCAL', 'category': 'Energy', 'drv': 2500.0},
        {'nutrient_id': 1062, 'name': 'Energy', 'unit_name': 'kJ', 'category': 'Energy', 'drv': 2500.0},
        {'nutrient_id': 1005, 'name': 'Carbohydrate', 'unit_name': 'G', 'category': 'Carbohydrates', 'drv': 320.0},
        {'nutrient_id': 1063, 'name': 'Sugars, Total', 'unit_name': 'G', 'category': 'Carbohydrates', 'drv': 50.0},
        {'nutrient_id': 1079, 'name': 'Fiber', 'unit_name': 'G', 'category': 'Carbohydrates', 'drv': 30.0},
        {'nutrient_id': 2000, 'name': 'Total Sugars', 'unit_name': 'G', 'category': 'Carbohydrates', 'drv': 50.0},
        {'nutrient_id': 2033, 'name': 'Fiber (AOAC 2011.25)', 'unit_name': 'G', 'category': 'Carbohydrates', 'drv': 28.0},
        {'nutrient_id': 1004, 'name': 'Total fat', 'unit_name': 'G', 'category': 'Lipids', 'drv': 80.0},
        {'nutrient_id': 1253, 'name': 'Cholesterol', 'unit_name': 'MG', 'category': 'Lipids', 'drv': 300.0},
        {'nutrient_id': 1258, 'name': 'Fatty acids, total saturated', 'unit_name': 'G', 'category': 'Lipids', 'drv': 20.0},
        {'nutrient_id': 1272, 'name': 'DHA', 'unit_name': 'G', 'category': 'Lipids', 'drv': 0.25},
        {'nutrient_id': 1278, 'name': 'EPA', 'unit_name': 'G', 'category': 'Lipids', 'drv': 0.25},
        {'nutrient_id': 1404, 'name': 'ALA', 'unit_name': 'G', 'category': 'Lipids', 'drv': 1.5},
        {'nutrient_id': 1003, 'name': 'Protein', 'unit_name': 'G', 'category': 'Proteins', 'drv': 160.0},
        {'nutrient_id': 1106, 'name': 'Vitamin A, RAE', 'unit_name': 'UG', 'category': 'Vitamins', 'drv': 900.0},
        {'nutrient_id': 1109, 'name': 'Vitamin E', 'unit_name': 'MG', 'category': 'Vitamins', 'drv': 15.0},
        {'nutrient_id': 1114, 'name': 'Vitamin D (D2 + D3)', 'unit_name': 'UG', 'category': 'Vitamins', 'drv': 20.0},
        {'nutrient_id': 1158, 'name': 'Vitamin E', 'unit_name': 'MG_ATE', 'category': 'Vitamins', 'drv': 15.0},
        {'nutrient_id': 1162, 'name': 'Vitamin C', 'unit_name': 'MG', 'category': 'Vitamins', 'drv': 90.0},
        {'nutrient_id': 1165, 'name': 'Thiamin', 'unit_name': 'MG', 'category': 'Vitamins', 'drv': 1.2},
        {'nutrient_id': 1166, 'name': 'Riboflavin', 'unit_name': 'MG', 'category': 'Vitamins', 'drv': 1.3},
        {'nutrient_id': 1167, 'name': 'Niacin', 'unit_name': 'MG', 'category': 'Vitamins', 'drv': 16.0},
        {'nutrient_id': 1170, 'name': 'Pantothenic acid', 'unit_name': 'MG', 'category': 'Vitamins', 'drv': 5.0},
        {'nutrient_id': 1175, 'name': 'Vitamin B-6', 'unit_name': 'MG', 'category': 'Vitamins', 'drv': 1.7},
        {'nutrient_id': 1177, 'name': 'Folate, total', 'unit_name': 'UG', 'category': 'Vitamins', 'drv': 400.0},
        {'nutrient_id': 1178, 'name': 'Vitamin B-12', 'unit_name': 'UG', 'category': 'Vitamins', 'drv': 2.4},
        {'nutrient_id': 1180, 'name': 'Choline', 'unit_name': 'MG', 'category': 'Vitamins', 'drv': 550.0},
        {'nutrient_id': 1183, 'name': 'Vitamin K2 MK-4', 'unit_name': 'UG', 'category': 'Vitamins', 'drv': 50.0},
        {'nutrient_id': 1185, 'name': 'Vitamin K1', 'unit_name': 'UG', 'category': 'Vitamins', 'drv': 120.0},
        {'nutrient_id': 2068, 'name': 'Vitamin E', 'unit_name': 'MG', 'category': 'Vitamins', 'drv': 15.0},
        {'nutrient_id': 2067, 'name': 'Vitamin A', 'unit_name': 'UG', 'category': 'Vitamins', 'drv': 900.0},
        {'nutrient_id': 1087, 'name': 'Calcium', 'unit_name': 'MG', 'category': 'Minerals', 'drv': 1000.0},
        {'nutrient_id': 1089, 'name': 'Iron', 'unit_name': 'MG', 'category': 'Minerals', 'drv': 10.0},
        {'nutrient_id': 1090, 'name': 'Magnesium', 'unit_name': 'MG', 'category': 'Minerals', 'drv': 420.0},
        {'nutrient_id': 1091, 'name': 'Phosphorus', 'unit_name': 'MG', 'category': 'Minerals', 'drv': 700.0},
        {'nutrient_id': 1092, 'name': 'Potassium', 'unit_name': 'MG', 'category': 'Minerals', 'drv': 3400.0},
        {'nutrient_id': 1093, 'name': 'Sodium', 'unit_name': 'MG', 'category': 'Minerals', 'drv': 1500.0},
        {'nutrient_id': 1095, 'name': 'Zinc', 'unit_name': 'MG', 'category': 'Minerals', 'drv': 11.0},
        {'nutrient_id': 1098, 'name': 'Copper', 'unit_name': 'MG', 'category': 'Minerals', 'drv': 0.9},
        {'nutrient_id': 1100, 'name': 'Iodine', 'unit_name': 'UG', 'category': 'Minerals', 'drv': 150.0},
        {'nutrient_id': 1101, 'name': 'Manganese', 'unit_name': 'MG', 'category': 'Minerals', 'drv': 2.3},
        {'nutrient_id': 1102, 'name': 'Molybdenum', 'unit_name': 'UG', 'category': 'Minerals', 'drv': 45.0},
        {'nutrient_id': 1103, 'name': 'Selenium', 'unit_name': 'UG', 'category': 'Minerals', 'drv': 55.0}
    ]
    return nutrients

def get_nutrients_with_drv_df():
    """
    Return a DataFrame of the nutrients with non-zero DRV values.
    """
    return pd.DataFrame(get_nutrients_with_drv()).rename(columns={"category": "nutrient_category", "name": "nutrientName"})
