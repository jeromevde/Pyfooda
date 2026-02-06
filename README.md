# FoodData Central Python API

This package provides a simple, ready-to-use Python API for accessing and querying relevant data from the [USDA FoodData Central](https://fdc.nal.usda.gov/) database—**no API key required**. All data is processed locally from CSV files and exposed through a clean Python interface.

## Installation

You can install the package from PyPI:

```bash
pip install pyfooda
```

Or install directly from the repository:

```bash
pip install git+https://github.com/yourusername/pyfooda.git
```

## Features

- **No API key required**: Works entirely offline with preprocessed FoodData Central CSV files.
- **Simple**: Query food categories, nutrients, and portion information with a few lines of code.
- **Search**: Find foods by partial name.
- **DataFrames**: Access the raw fooddata and DRV DataFrames for custom analysis.
- **🆕 Agentic Aggregation**: Merge similar food items into a lightweight database using AI-powered workflow (see [AGGREGATION_PLAN.md](AGGREGATION_PLAN.md))

## Example

```python
import pyfooda as pf

# Find up to 10 foods matching a partial name
print(pf.find_closest_matches('apple'))

# Get the category of a food
print(pf.get_category('Apple, raw'))

# Get all nutrient values for a food
nutrients = pf.get_nutrients('Apple, raw')
print(nutrients)

# Get portion information
print(pf.get_portion_gram_weight('Apple, raw'))  # e.g., 138.0
print(pf.get_portion_unit_name('Apple, raw'))    # e.g., "medium"

# Get the raw DataFrames
fooddata_df = pf.get_fooddata_df()
drv_df = pf.get_drv_df()
```

## API Reference

### `get_category(foodName)`
Returns the food category for the given food name (case-insensitive). Returns `'Other'` if not found.

### `get_nutrients(foodName)`
Returns a dictionary of nutrient values for the given food name. Returns `None` if not found.

### `get_portion_gram_weight(foodName)`
Returns the portion gram weight (float) for the given food name. Returns `None` if not found.

### `get_portion_unit_name(foodName)`
Returns the portion unit name (string) for the given food name. Returns `None` if not found.

### `find_closest_matches(partialName)`
Returns a list of up to 10 food names that contain the given partial name (case-insensitive).

### `get_fooddata_df()`
Returns the fooddata DataFrame containing all food items and their nutrient values.

### `get_drv_df()`
Returns the DRV (Dietary Reference Values) DataFrame containing nutrient reference values.

## Food Aggregation (New!)

The current database has 295,943 food items, which is overwhelming for everyday use. The new **agentic aggregation workflow** intelligently merges similar items into a lightweight database.

### Quick Start
```bash
# Install dependencies
pip install -r requirements_aggregation.txt

# Run on 1000-item sample (no API key needed)
python aggregate_foods.py --sample 1000

# Or with LLM validation for better quality
export OPENROUTER_API_KEY="your-key"
python aggregate_foods.py --sample 1000 --use-llm
```

See [AGGREGATION_PLAN.md](AGGREGATION_PLAN.md) for full documentation.

## License

## Test package

```
pip install -e .
python -c "import pyfooda; print(pyfooda.__version__)"
```

MIT License
