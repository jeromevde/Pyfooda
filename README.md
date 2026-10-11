# Pyfooda

Offline nutrition lookup for recipe ingredients.

We first tried **LLM aggregation** over raw USDA rows — grouping ~300k branded food names into generic foods with an LLM. It worked on a test set but did not scale: noisy names, duplicate groups, and fragile matching.

We switched to the **[Epicure](https://huggingface.co/datasets/Kaikaku/epicure-corpus-resources) vocabulary** (1,790 canonical recipe ingredients) and map USDA rows to those slots with embeddings, averaging up to 5 agreeing matches per ingredient.

```bash
pip install pyfooda
```

```python
import pyfooda as pf

pf.find_ingredients("hazelnut")
pf.get_nutrients("chocolate_hazelnut_spread")
pf.get_matches("chocolate_hazelnut_spread")
pf.list_clusters()           # food families for browsing
pf.get_cluster("vegetables")
pf.get_nutrient_coverage()   # database-wide: which nutrients are measured
pf.get_coverage("almond")    # per-ingredient: measured vs missing
```

**Web UI:** [jeromevde.github.io/Pyfooda](https://jeromevde.github.io/Pyfooda/) — search the catalog, or open a food family and click through to nutrients and the USDA rows behind them.

## Rebuild from source

One command reproduces the published table from the stored USDA matches and `fooddata.csv.gz`. It does not call an embedding model or a language model.

```bash
pip install -e .
pyfooda-rebuild
```

That aligns each ingredient's USDA rows, recomputes nutrients, and refreshes `docs/data/ingredients.json`.

`pyfooda-build` is the older full rematch. It needs the `build` extra (`sentence-transformers`) and replaces the stored matches.

Data: Epicure vocabulary (CC BY 4.0) + USDA FoodData Central (public domain).
