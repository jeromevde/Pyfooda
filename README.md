# Pyfooda

Offline nutrition lookup for recipe ingredients.

We first tried **LLM aggregation** over raw USDA rows — grouping ~300k branded food names into generic foods with an LLM. It worked on a test set but did not scale: noisy names, duplicate groups, and fragile matching.

We switched to the **[Epicure](https://huggingface.co/datasets/Kaikaku/epicure-corpus-resources) vocabulary** (1,790 canonical recipe ingredients) and map USDA rows to those slots with embeddings, averaging up to 5 close matches per ingredient.

```bash
pip install pyfooda
```

```python
import pyfooda as pf

pf.find_ingredients("hazelnut")
pf.get_nutrients("chocolate_hazelnut_spread")
pf.get_sources("chocolate_hazelnut_spread")
```

**Web UI:** [jeromevde.github.io/Pyfooda](https://jeromevde.github.io/Pyfooda/)

**Rebuild:** `pip install -r requirements-build.txt && python scripts/build_database.py`

Data: Epicure vocabulary (CC BY 4.0) + USDA FoodData Central (public domain).
