#!/usr/bin/env python3
"""
Food Aggregation Pipeline

Collapses USDA FoodData Central entries into a compact generic food database
using an LLM.  Processes foods in batches; the LLM sees each item's nearest
existing generic entries and decides CREATE / ADD / IGNORE.

Usage:
  python scripts/run_aggregation.py --mode test --batch-size 100
  python scripts/run_aggregation.py --mode full --batch-size 100 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import openai

DEFAULT_BATCH_SIZE = 100


# Lazy-loaded embedding model (shared across instances, loaded once)
_embedding_model = None


def _get_embedding_model():
    """Load the sentence-transformer model once on first use."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _embedding_model


# ---------------------------------------------------------------------------
# Search tool — embedding-based (sentence-transformers + faiss)
# ---------------------------------------------------------------------------


class FoodSearchIndex:
    """Embedding-based search index over aggregated food names.

    Uses BAAI/bge-small-en-v1.5 for dense embeddings and faiss for
    fast approximate nearest-neighbor search.

    Keeps the same interface as the old BM25/trigram index:
      add(food_id, name)
      search(query, top_k) -> [{id, name, score}]
      rename(food_id, new_name)
    """

    def __init__(self):
        import faiss

        self.names: list[str] = []
        self.ids: list[int] = []
        self._model = _get_embedding_model()
        self._dim = self._model.get_sentence_embedding_dimension()
        self._faiss_index = faiss.IndexFlatIP(self._dim)  # inner-product (cosine after normalization)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode text(s) into L2-normalized embeddings."""
        import faiss
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.ascontiguousarray(vecs.astype('float32'))

    def add(self, food_id: int, name: str):
        vec = self._embed([name.lower()])
        self.names.append(name.lower())
        self.ids.append(food_id)
        self._faiss_index.add(vec)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top-k closest entries as [{id, name, score}]."""
        if not self.names:
            return []

        q_vec = self._embed([query.lower()])
        k = min(top_k, len(self.names))
        scores, indices = self._faiss_index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            cos_sim = float(score)  # already cosine similarity (normalized IP)
            if cos_sim > 0.35:      # minimum relevance threshold
                results.append({
                    'id': self.ids[idx],
                    'name': self.names[idx],
                    'score': round(cos_sim, 3),
                })
        return results

    def rename(self, food_id: int, new_name: str):
        """Update the name of an existing entry in the index.

        faiss IndexFlatIP doesn't support in-place updates, so we
        rebuild the full index (cheap at <100k entries).
        """
        try:
            pos = self.ids.index(food_id)
        except ValueError:
            return
        self.names[pos] = new_name.lower()
        # Rebuild index
        self._rebuild()

    def _rebuild(self):
        """Re-encode all names and rebuild the faiss index from scratch."""
        import faiss
        self._faiss_index = faiss.IndexFlatIP(self._dim)
        if self.names:
            vecs = self._embed(self.names)
            self._faiss_index.add(vecs)

    def __len__(self):
        return len(self.names)


# ---------------------------------------------------------------------------
# Nutrient helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Data-type priority — foundation foods are the gold standard
# ---------------------------------------------------------------------------

_DATA_TYPE_PRIORITY = {
    'foundation_food': 0,
    'sr_legacy_food': 1,
    'survey_fndds_food': 1,
    'sub_sample_food': 2,
    'agricultural_acquisition': 2,
    'branded_food': 3,
}


def _data_type_rank(dtype: str) -> int:
    """Return sort rank for a data_type value (lower = higher priority)."""
    return _DATA_TYPE_PRIORITY.get(str(dtype).strip(), 9)


NUTRIENT_COLS = [
    'Energy', 'Carbohydrate', 'Protein', 'Total fat',
    'Fiber', 'Sugars, Total', 'Sodium', 'Calcium', 'Iron',
    'Cholesterol', 'Fatty acids, total saturated',
    'Vitamin A, RAE', 'Vitamin C', 'Vitamin D (D2 + D3)',
    'Vitamin E', 'Vitamin B-6', 'Vitamin B-12',
    'Folate, total', 'Zinc', 'Magnesium', 'Potassium', 'Phosphorus',
]


def _avg_nutrients(existing: dict, new_row: pd.Series, count: int) -> dict:
    """Running average of nutrients. count is the current count before adding."""
    merged = {}
    for col in NUTRIENT_COLS:
        old_val = existing.get(col)
        new_val = new_row.get(col, np.nan)
        old_na = old_val is None or (isinstance(old_val, float) and np.isnan(old_val))
        new_na = pd.isna(new_val)
        if old_na and new_na:
            merged[col] = None
        elif old_na:
            merged[col] = float(new_val)
        elif new_na:
            merged[col] = float(old_val)
        else:
            merged[col] = (float(old_val) * count + float(new_val)) / (count + 1)
    return merged


# Key nutrients shown in prompt (compact fingerprint for LLM comparison)
_PROMPT_NUTRIENT_COLS = ['Energy', 'Carbohydrate', 'Protein', 'Total fat', 'Sugars, Total']
_PROMPT_NUTRIENT_SHORT = ['kcal', 'carb', 'prot', 'fat', 'sugar']


def _nutrient_fingerprint(row_or_dict, is_dict: bool = False) -> str:
    """Compact nutrient string like 'kcal=34 carb=9 prot=0 fat=0 sugar=8'."""
    parts = []
    for col, short in zip(_PROMPT_NUTRIENT_COLS, _PROMPT_NUTRIENT_SHORT):
        if is_dict:
            val = row_or_dict.get(col)
        else:
            val = row_or_dict.get(col, np.nan)
            if pd.isna(val):
                val = None
        if val is not None:
            parts.append(f'{short}={float(val):.0f}')
        else:
            parts.append(f'{short}=?')
    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT_PATH = Path(__file__).parent / 'aggregation_prompt.txt'


def load_prompt(path: Optional[str] = None) -> str:
    p = Path(path) if path else _DEFAULT_PROMPT_PATH
    return p.read_text()


def _energy_from_fingerprint(fp: str) -> float | None:
    """Extract kcal value from a nutrient fingerprint string."""
    m = re.search(r'kcal=(\d+)', fp)
    return float(m.group(1)) if m else None


def _build_item_prompt(food: dict) -> str:
    """Build a single-item prompt.

    Streaming mode: each LLM call handles exactly one incoming food.
    """
    sr = food['search_results']
    incoming_energy = _energy_from_fingerprint(food.get('nutrients_str', ''))

    # Track valid ADD target ids for guard checks
    food['_valid_add_ids'] = {m['id'] for m in sr}

    if not sr:
        matches_str = '  (no existing entries yet — use CREATE only, not ADD)'
    else:
        match_lines = []
        for m in sr:
            m_energy = _energy_from_fingerprint(m.get('nutrients', ''))
            warn = ''
            kcal_diff_str = ''
            if incoming_energy and m_energy and incoming_energy > 0 and m_energy > 0:
                ratio = abs(incoming_energy - m_energy) / min(incoming_energy, m_energy)
                pct = int(ratio * 100)
                if ratio > 0.40:
                    warn = f'  [BLOCK: kcal diff={pct}% — likely different foods, use CREATE]'
                elif ratio > 0.20:
                    kcal_diff_str = f'  [kcal diff={pct}%]'
            count = m.get('count', 1)
            count_str = f'  ({count} sources)' if count > 1 else ''
            # Show up to 3 representative source names so the model can verify
            # the group actually contains the same specific food
            examples = m.get('source_names', [])[:3]
            examples_str = f'  e.g.: {"; ".join(examples)}' if examples else ''
            cat_str = f'  [cat={m["food_category"]}]' if m.get("food_category") else ''
            match_lines.append(
                f'    id={m["id"]}  "{m["name"]}"  [{m.get("nutrients", "")}]{cat_str}  (score={m["score"]}){count_str}{kcal_diff_str}{examples_str}{warn}'
            )
        matches_str = '\n'.join(match_lines)

    nutrients_str = food.get('nutrients_str', '')
    return (
        f'[{food["idx"]}] "{food["name"]}"  category={food["category"]}  [{nutrients_str}]\n'
        f'Closest existing entries (ADD id must come from this list ONLY):\n{matches_str}'
    )


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

COPILOT_PROXY_BASE_URL = 'http://127.0.0.1:3000/v1'
COPILOT_PROXY_MODEL = 'gpt-5-mini'


def _resolve_llm_backend(
    api_key: str | None,
    base_url: str | None,
    model: str,
) -> tuple[str, str, str]:
    """Return (api_key, base_url, model) resolving the active LLM backend.

    Priority:
      1. Explicit args passed by caller.
      2. OPENROUTER_API_KEY env var  → OpenRouter.
      3. Fallback: VS Code Copilot proxy at 127.0.0.1:3000
         (https://github.com/hyorman/copilot-proxy — run locally).
    """
    resolved_key = api_key or os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
    resolved_url = base_url or os.getenv('OPENROUTER_BASE_URL')
    resolved_model = model

    if resolved_key and resolved_url:
        # Fully explicit — use as-is.
        return resolved_key, resolved_url, resolved_model

    if resolved_key:
        # Key but no URL → OpenRouter default.
        return resolved_key, 'https://openrouter.ai/api/v1', resolved_model

    # No key at all → fall back to local Copilot proxy.
    # Use the caller-supplied model if given, else the proxy default.
    proxy_model = model if model else COPILOT_PROXY_MODEL
    print(
        '  OPENROUTER_API_KEY not set — falling back to VS Code Copilot proxy '
        f'at {COPILOT_PROXY_BASE_URL}  (model: {proxy_model})'
    )
    return 'dummy', COPILOT_PROXY_BASE_URL, proxy_model


def _call_llm_batch(
    system_prompt: str,
    user_message: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.05,
    max_retries: int = 3,
    timeout_seconds: int = 600,
) -> str:
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_message},
                ],
                temperature=temperature,
                max_tokens=16384,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f'  LLM error ({e}), retrying in {wait}s ...')
                time.sleep(wait)
            else:
                raise


def _parse_llm_decision(raw: str, expected_idx: int) -> dict | None:
    """Parse one decision for one item.

    Accepted forms:
      [idx] CREATE <name>
      [idx] ADD <id_or_name>
      [idx] IGNORE
    """
    # JSON support
    try:
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = '\n'.join(cleaned.split('\n')[1:])
        if cleaned.endswith('```'):
            cleaned = '\n'.join(cleaned.split('\n')[:-1])
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and 'action' in parsed:
            parsed.setdefault('idx', expected_idx)
            parsed['action'] = str(parsed['action']).upper()
            return parsed
        if isinstance(parsed, dict) and 'decision' in parsed:
            raw = str(parsed['decision'])
        if isinstance(parsed, list) and parsed:
            cand = parsed[0]
            if isinstance(cand, dict) and 'action' in cand:
                cand.setdefault('idx', expected_idx)
                cand['action'] = str(cand['action']).upper()
                return cand
    except (json.JSONDecodeError, ValueError):
        pass

    for line in raw.strip().splitlines():
        line = line.strip().lstrip('-').strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue

        # Accept no-index variant by prefixing expected idx.
        if re.match(r'^(CREATE|ADD|IGNORE)\b', line, re.IGNORECASE):
            line = f'[{expected_idx}] ' + line

        m = re.match(r'\[?(\d+)\]?\s*(CREATE|ADD|IGNORE)\s*(.*)', line, re.IGNORECASE)
        if not m:
            continue
        action = m.group(2).upper()
        rest = m.group(3).strip()

        if action == 'CREATE':
            name = rest.strip().strip('"').strip("'").strip(',').strip()
            return {'idx': expected_idx, 'action': 'CREATE', 'name': name or 'Unknown Food'}
        if action == 'ADD':
            rest_clean = rest.strip().strip('"').strip("'")
            try:
                return {'idx': expected_idx, 'action': 'ADD', 'target_id': int(rest_clean.split()[0])}
            except (ValueError, IndexError):
                return {'idx': expected_idx, 'action': 'ADD', 'target_name': rest_clean}
        if action == 'IGNORE':
            return {'idx': expected_idx, 'action': 'IGNORE'}

    return None


# ---------------------------------------------------------------------------
# Main aggregation engine
# ---------------------------------------------------------------------------


class FoodAggregator:
    """
    Processes the full USDA foods list and builds a compact aggregated DB.

    Usage:
        agg = FoodAggregator(foods_df)
        run_batched(agg, limit=1000, offset=0, resume_from=0, batch_size=24)
        agg.save('output.json')
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        model: str = 'google/gemini-2.0-flash-lite-001',
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        batch_size: int = 50,
        search_top_k: int = 8,
        checkpoint_dir: str = './checkpoints',
        timeout_seconds: int = 600,
    ):
        self.source_df = df.copy().reset_index(drop=True)
        self.api_key, self.base_url, self.model = _resolve_llm_backend(api_key, base_url, model)
        self.system_prompt = system_prompt or DEFAULT_PROMPT
        self.batch_size = batch_size
        self.search_top_k = search_top_k
        self.timeout_seconds = int(timeout_seconds)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Aggregated database  {id -> entry}
        self.db: dict[int, dict] = {}
        self._next_id = 1
        self.index = FoodSearchIndex()
        # Exact name → id lookup (lowercase) for duplicate-name dedup
        self._name_to_id: dict[str, int] = {}

        # Stats
        self.stats = {'created': 0, 'added': 0, 'ignored': 0, 'errors': 0, 'api_calls': 0,
                      'foundation_locked': 0}
        self.processed_count = 0

    # -- actions ----------------------------------------------------------------

    def _do_create(self, generic_name: str, row: pd.Series):
        fid = self._next_id
        self._next_id += 1

        nutrients = {}
        for col in NUTRIENT_COLS:
            val = row.get(col, np.nan)
            nutrients[col] = float(val) if pd.notna(val) else None

        # Collect serving size if available
        pgw = row.get('portion_gram_weight', np.nan)
        portion_weights = [float(pgw)] if pd.notna(pgw) else []
        pun = row.get('portion_unit_name', '')
        portion_units = [str(pun)] if pd.notna(pun) and str(pun).strip() else []

        # Track source energies for outlier gating
        energy_val = row.get('Energy', np.nan)
        source_energies = [float(energy_val)] if pd.notna(energy_val) else []

        dtype = str(row.get('data_type', ''))
        self.db[fid] = {
            'id': fid,
            'generic_name': generic_name,
            'food_category': str(row.get('food_category', '')),
            'nutrients': nutrients,
            'source_ids': [int(row.name) if isinstance(row.name, (int, np.integer)) else 0],
            'source_names': [str(row.get('foodName', ''))],
            'portion_gram_weights': portion_weights,
            'portion_unit_names': portion_units,
            '_source_energies': source_energies,
            '_created_by': dtype,
            'count': 1,
        }
        self.index.add(fid, generic_name)
        self._name_to_id[generic_name.strip().lower()] = fid

    def _do_add(self, target_id: int, row: pd.Series) -> str:
        """Add a source food to an existing group. Returns 'ok'."""
        entry = self.db[target_id]
        entry['nutrients'] = _avg_nutrients(entry['nutrients'], row, entry['count'])
        entry['count'] += 1
        src_id = int(row.name) if isinstance(row.name, (int, np.integer)) else 0
        entry['source_ids'].append(src_id)
        entry['source_names'].append(str(row.get('foodName', '')))
        pgw = row.get('portion_gram_weight', np.nan)
        if pd.notna(pgw):
            entry['portion_gram_weights'].append(float(pgw))
        pun = row.get('portion_unit_name', '')
        if pd.notna(pun) and str(pun).strip():
            entry['portion_unit_names'].append(str(pun))
        return 'ok'

    def _apply_similarity_merge_postpass(self, **_):
        """No-op — postpass removed."""

    def _apply_canonical_postpass(self, **_):
        """No-op — postpass removed."""

    # -- persistence ------------------------------------------------------------

    def _save_checkpoint(self):
        path = self.checkpoint_dir / 'aggregation_checkpoint.json'
        # Use the same {meta, foods} format as the final output so the
        # checkpoint is directly inspectable / usable as an output file.
        foods = list(self.db.values())            # keep _* internal fields
        meta = {
            'processed_count': self.processed_count,
            'next_id': self._next_id,
            'stats': self.stats,
        }
        with open(path, 'w') as f:
            json.dump({'meta': meta, 'foods': foods}, f, indent=2, default=str)

    def load_checkpoint(self) -> bool:
        path = self.checkpoint_dir / 'aggregation_checkpoint.json'
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)

        # Support both new {meta, foods} format and legacy {db, ...} format
        if 'meta' in data and 'foods' in data:
            meta = data['meta']
            self.processed_count = meta['processed_count']
            self._next_id = meta['next_id']
            self.stats = meta['stats']
            self.db = {entry['id']: entry for entry in data['foods']}
        else:
            # Legacy format
            self.processed_count = data['processed_count']
            self._next_id = data['next_id']
            self.stats = data['stats']
            self.db = {int(k): v for k, v in data['db'].items()}

        self.index = FoodSearchIndex()
        self._name_to_id = {}
        for fid, entry in self.db.items():
            self.index.add(fid, entry['generic_name'])
            self._name_to_id[entry['generic_name'].strip().lower()] = fid
        print(
            f'  Resumed from checkpoint: {self.processed_count} items processed, '
            f'{len(self.db)} generic foods'
        )
        return True

    # -- output -----------------------------------------------------------------

    def save(self, output_path: str = 'pyfooda/data/foods_aggregated.json'):
        """Save the compact aggregated database (same format as checkpoint)."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Strip internal fields before saving
        clean = []
        for entry in self.db.values():
            e = {k: v for k, v in entry.items() if not k.startswith('_')}
            clean.append(e)

        meta = {
            'processed_count': self.processed_count,
            'stats': self.stats,
        }
        with open(out, 'w') as f:
            json.dump({'meta': meta, 'foods': clean}, f, indent=2, default=str)
        print(f'  Saved {len(self.db)} generic foods -> {output_path}')

        csv_path = str(out).replace('.json', '.csv')
        rows = []
        for entry in self.db.values():
            # Compute median serving size
            pw = entry.get('portion_gram_weights', [])
            median_gw = float(np.median(pw)) if pw else None
            # Most common portion unit name
            pu = entry.get('portion_unit_names', [])
            portion_unit = max(set(pu), key=pu.count) if pu else None

            row = {
                'foodName': entry['generic_name'],
                'category': entry['food_category'],
                'portion_gram_weight': median_gw,
                'portion_unit_name': portion_unit,
            }
            if entry['nutrients']:
                row.update(entry['nutrients'])
            rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f'  Saved CSV -> {csv_path}')

    def _print_summary(self):
        print()
        print('=' * 60)
        print('  AGGREGATION COMPLETE')
        print('=' * 60)
        print(f'  Source foods processed : {self.processed_count}')
        print(f'  Generic foods created  : {self.stats["created"]}')
        print(f'  Merged into existing   : {self.stats["added"]}')
        print(f'  Ignored                : {self.stats["ignored"]}')
        print(f'  Foundation-locked (skip): {self.stats.get("foundation_locked", 0)}')
        print(f'  ADD id rejected (bad idx): {self.stats.get("add_id_rejected", 0)}')
        print(f'  Name dedup merged       : {self.stats.get("name_dedup_merged", 0)}')
        print(f'  Parse errors (-> create) : {self.stats["errors"]}')
        print(f'  API calls              : {self.stats["api_calls"]}')
        print(f'  Final DB size          : {len(self.db)} entries')
        print(f'  Similarity merged groups: {self.stats.get("similarity_merged_groups", 0)}')
        print(f'  Post-pass merged groups: {self.stats.get("postpass_merged_groups", 0)}')
        print('=' * 60)

DEFAULT_PROMPT = """You are building a compact food reference database from USDA FoodData Central.

For each food item, output one decision using the JSON format shown in the batch instruction.

NAMING — write short 2-4 word Title Case generics, never raw USDA strings:
  "Biscuits, plain or buttermilk, dry mix"      → "Biscuits Dry Mix"
  "Fish, salmon, sockeye, wild caught, raw"      → "Wild Salmon Raw"
  "Game meat, deer, raw"                          → "Deer Raw"  (use source word: "Deer" not "Venison")
  "Sauce, pesto, ready-to-serve, refrigerated"   → "Pesto Sauce"
  "Nuts, almonds, dry roasted, with salt added"  → "Almonds Dry Roasted"
  No lot IDs, no lab codes, no moisture strings (e.g. "11F 8119", "NFS", "NS as to" are invalid).

MERGE (ADD) these variant types into the base food group:
  • Brand variants: "BUD LIGHT" → Beer Light;  "BUDWEISER" → Beer Regular
  • Fortified/enriched: "grape juice with calcium" → Grape Juice
  • Reduced-sodium / no-salt-added (when kcal matches)
  • Toast/heat variants: "French bread toasted" → French Bread
  • Baby food stages: Stage 1 + Stage 2 of same food → one group
  • Wine varietals of same color: Merlot, Syrah, Claret → "Red Table Wine"
  • Refrigerated vs shelf-stable of same dish: pesto refrigerated + shelf stable → "Pesto Sauce"

KEEP SEPARATE — always CREATE a new group when:
  • Cooking state differs: raw ≠ cooked ≠ canned ≠ frozen ≠ dried
  • Fat tier differs: fat-free ≠ reduced-fat ≠ regular  (fat-free has ~90% fewer kcal — check kcal!)
  • Dry form ≠ hydrated/cooked (pasta dry 370 kcal ≠ pasta cooked 130 kcal)
  • The candidate's example sources show a DIFFERENT food (almonds ≠ chestnuts; broth ≠ sauce)
  • Cheese varieties (Mozzarella ≠ Cheddar), grain types, pizza styles
"""


def _normalized_name_key(name: str) -> str:
    import re
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_generic_name(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name)).strip()

    # Strip lab/spec tokens: NFS, NFs, NS as to, lot IDs, moisture % strings
    s = re.sub(r'\bNFS\b', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\bNFs\b', '', s).strip()
    s = re.sub(r'\bNS\s+as\s+to\b.*', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\s+', ' ', s).strip()

    # hard cap to reduce overly specific labels
    words = s.split()
    if len(words) > 6:
        s = " ".join(words[:6])

    return s


def _find_existing_id_by_name(agg: FoodAggregator, candidate: str):
    key = _normalized_name_key(candidate)
    # fast path: exact mapping
    if key in agg._name_to_id:
        return agg._name_to_id[key]
    # fallback scan to catch legacy keys
    for k, v in agg._name_to_id.items():
        if _normalized_name_key(k) == key:
            return v
    return None


def _resolve_input_output(repo_root: Path, mode: str, output: str | None, checkpoint_dir: str | None):
    default_input = repo_root / "pyfooda/data/fooddata.csv"
    default_output = repo_root / "scripts/output/batch_test_aggregated.json" if mode == "test" else repo_root / "pyfooda/data/foods_aggregated_batch.json"
    default_ckpt = repo_root / ("scripts/output/checkpoints_batch" if mode == "test" else "checkpoints_batch")

    # In test mode, use the purpose-built test_fooddata.csv if present — it
    # contains exactly the foods referenced in test_set.json, so scoring works
    # correctly without needing to process the entire dataset.
    # Regenerate it with: python scripts/generate_test_csv.py
    input_path = default_input
    if mode == "test":
        test_csv = repo_root / "scripts/test_fooddata.csv"
        if test_csv.exists():
            input_path = test_csv

    return input_path, Path(output) if output else default_output, Path(checkpoint_dir) if checkpoint_dir else default_ckpt





def _apply_decision(agg: FoodAggregator, food: dict, row: pd.Series, decision: dict, create_aliases: dict[str, str] | None = None):
    name = food["name"]
    if decision["action"] == "CREATE":
        create_name = _sanitize_generic_name(decision.get("name", name))
        if create_aliases is not None:
            key = _normalized_name_key(create_name)
            create_name = create_aliases.setdefault(key, create_name)
        existing_id = _find_existing_id_by_name(agg, create_name)
        if existing_id is not None:
            agg._do_add(existing_id, row)
            agg.stats["added"] += 1
        else:
            agg._do_create(create_name, row)
            agg._name_to_id[_normalized_name_key(create_name)] = agg._next_id - 1
            agg.stats["created"] += 1

    elif decision["action"] == "ADD":
        target = decision.get("target_id")
        target_name = decision.get("target_name")
        valid_add_ids = food.get("_valid_add_ids", set())

        if target is not None and target not in valid_add_ids:
            target = None

        if target is None and target_name:
            # First: resolve within-batch create_aliases (group just created in this batch)
            if create_aliases:
                key = _normalized_name_key(target_name)
                canonical = create_aliases.get(key)
                if canonical is not None:
                    target = _find_existing_id_by_name(agg, canonical)
            # Fallback: FAISS search among already-indexed groups
            if target is None:
                matches = agg.index.search(target_name, top_k=1)
                if matches and matches[0]["score"] > 0.75 and matches[0]["id"] in valid_add_ids:
                    target = matches[0]["id"]

        if target and target in agg.db:
            agg._do_add(target, row)
            agg.stats["added"] += 1
        else:
            cname = _sanitize_generic_name(target_name or name)
            agg._do_create(cname, row)
            agg._name_to_id[_normalized_name_key(cname)] = agg._next_id - 1
            agg.stats["created"] += 1

    elif decision["action"] == "IGNORE":
        agg.stats["ignored"] += 1
    else:
        agg.stats["errors"] += 1
        agg.stats["ignored"] += 1


_DEFAULT_BATCH_INSTRUCTION = (
    "\n\nReturn exactly one decision per item as strict JSON object lines."
    "\nAllowed JSON schemas:"
    "\n{\"idx\": <idx>, \"action\": \"CREATE\", \"name\": \"<generic_name>\"}"
    "\n{\"idx\": <idx>, \"action\": \"ADD\", \"target_id\": <id>}"
    "\n{\"idx\": <idx>, \"action\": \"ADD\", \"target_name\": \"<generic_name>\"}"
    "\n{\"idx\": <idx>, \"action\": \"IGNORE\"}"
    "\n"
    "\nRules:"
    "\n- target_id must come from that item's candidate list."
    "\n  target_name references a group CREATEd earlier in THIS same batch — use this for within-batch variants/brands."
    "\n- Within-batch merging: before writing CREATE, check if an earlier item in this batch already CREATEd the right"
    " generic group. If so, ADD to it using target_name instead of creating a duplicate."
    "\n- ADD only if the candidate's example sources confirm it is the SAME food."
    " Check category: 'Dairy' ≠ 'Finfish'; different nuts (almonds ≠ chestnuts) are NEVER merged."
    "\n- When in doubt, CREATE with a short 2-4 word Title Case name."
)


def run_batched(
    agg: FoodAggregator,
    *,
    limit: int | None,
    offset: int,
    resume_from: int,
    batch_size: int,
    batch_instruction: str | None = None,
    on_batch_complete=None,
):
    df = agg.source_df
    df = df[df["foodName"].notna() & (df["foodName"].astype(str) != "nan")].reset_index(drop=True)
    df["_dtype_rank"] = df["data_type"].apply(_data_type_rank)
    df = df.sort_values(["_dtype_rank", "food_category"], ascending=[True, True], na_position="last").reset_index(drop=True)
    if offset:
        df = df.iloc[offset:].reset_index(drop=True)
    if limit:
        df = df.iloc[:limit]

    total = len(df)
    start_ts = time.time()

    i = resume_from
    while i < total:
        chunk = []
        end = min(i + batch_size, total)
        for idx in range(i, end):
            row = df.iloc[idx]
            name = str(row.get("foodName", ""))
            cat = str(row.get("food_category", ""))
            raw_results = agg.index.search(name, top_k=agg.search_top_k)
            search_results = []
            for m in raw_results:
                entry = agg.db.get(m["id"])
                if entry:
                    m["nutrients"] = _nutrient_fingerprint(entry["nutrients"], is_dict=True)
                    m["count"] = entry.get("count", 1)
                    m["source_names"] = entry.get("source_names", [])[:3]
                    m["food_category"] = entry.get("food_category", "")
                    search_results.append(m)

            food = {
                "idx": idx,
                "name": name,
                "category": cat,
                "nutrients_str": _nutrient_fingerprint(row),
                "search_results": search_results,
                "_valid_add_ids": {m["id"] for m in search_results},
            }
            chunk.append((idx, row, food))

        user_msg = "\n\n".join(_build_item_prompt(food) for _, _, food in chunk)
        user_msg += batch_instruction if batch_instruction is not None else _DEFAULT_BATCH_INSTRUCTION

        batch_num = (i - resume_from) // batch_size + 1
        total_batches = (total - resume_from + batch_size - 1) // batch_size
        elapsed = time.time() - start_ts
        groups_so_far = len(agg.db)
        print(f"  Batch {batch_num}/{total_batches}  rows {i+1}-{end}/{total}  groups={groups_so_far}  elapsed={elapsed:.0f}s", flush=True)

        try:
            raw = _call_llm_batch(
                agg.system_prompt,
                user_msg,
                api_key=agg.api_key,
                base_url=agg.base_url,
                model=agg.model,
                timeout_seconds=agg.timeout_seconds,
            )
            agg.stats["api_calls"] += 1
        except Exception:
            # fallback: mark all as ignored on batch failure
            for _idx, _row, _food in chunk:
                agg.stats["errors"] += 1
                agg.stats["ignored"] += 1
                agg.processed_count += 1
            i = end
            continue

        # Map each response line to its idx
        line_map: dict[int, str] = {}
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict) and obj.get('idx') is not None:
                    line_map[int(obj['idx'])] = ln
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        create_aliases: dict[str, str] = {}
        for idx, row, food in chunk:
            decision = _parse_llm_decision(line_map[idx], idx) if idx in line_map else None
            if decision is None:
                decision = _parse_llm_decision(raw, idx)
            if decision is None:
                agg.stats["errors"] += 1
                agg.stats["ignored"] += 1
            else:
                _apply_decision(agg, food, row, decision, create_aliases=create_aliases)
            agg.processed_count += 1

        if on_batch_complete is not None:
            on_batch_complete([food for _, _, food in chunk])

        if agg.processed_count % 500 == 0:
            agg._save_checkpoint()
        i = end

    agg._apply_similarity_merge_postpass(threshold=0.80)
    agg._apply_canonical_postpass()
    agg._save_checkpoint()
    agg._print_summary()

    elapsed = time.time() - start_ts
    return {"elapsed_seconds": elapsed, "processed": agg.processed_count, "api_calls": agg.stats.get("api_calls", 0), "final_groups": len(agg.db)}


def main():
    parser = argparse.ArgumentParser(description="Run aggregation pipeline (batched)")
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    parser.add_argument("--model", default=None,
                        help="Model name. Defaults to gpt-5-mini via Copilot proxy when no API key is set.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--search-top-k", type=int, default=8)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="skip first N sorted rows before processing")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--estimate-full-size", type=int, default=296000)
    parser.add_argument("--estimated-cost-per-call", type=float, default=0.0, help="optional rough USD estimate per LLM API call")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path, output_path, checkpoint_dir = _resolve_input_output(repo_root, args.mode, args.output, args.checkpoint_dir)

    # Resolve LLM backend — falls back to local Copilot proxy if no key is set.
    args.api_key, args.base_url, args.model = _resolve_llm_backend(
        args.api_key, args.base_url, args.model
    )

    df = pd.read_csv(input_path)
    system_prompt = Path(args.prompt).read_text() if args.prompt else DEFAULT_PROMPT

    agg = FoodAggregator(
        df,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        batch_size=args.batch_size,
        search_top_k=args.search_top_k,
        system_prompt=system_prompt,
        checkpoint_dir=str(checkpoint_dir),
        timeout_seconds=args.timeout_seconds,
    )

    resume_from = 0
    if args.resume and agg.load_checkpoint():
        resume_from = agg.processed_count

    limit = args.limit
    if args.mode == "full":
        limit = limit or None

    result = run_batched(
        agg,
        limit=limit,
        offset=args.offset,
        resume_from=resume_from,
        batch_size=args.batch_size,
    )
    agg.save(str(output_path))

    # Human-review friendly grouping trace: Group Name: item1, item2, ...
    groups_txt = output_path.with_suffix('.groups.txt')
    lines = []
    for entry in sorted(agg.db.values(), key=lambda e: str(e.get('generic_name', '')).lower()):
        gname = entry.get('generic_name', 'Unknown Group')
        src_names = entry.get('source_names', [])
        unique_src = []
        seen = set()
        for s in src_names:
            t = str(s).strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            unique_src.append(t)
        lines.append(f"{gname}: " + " | ".join(unique_src))
    groups_txt.write_text("\n".join(lines) + "\n")

    rate = result["processed"] / result["elapsed_seconds"] if result["elapsed_seconds"] > 0 else 0
    eta_seconds = args.estimate_full_size / rate if rate > 0 else math.inf

    est_cost = result["api_calls"] * args.estimated_cost_per_call
    est_full_calls = (args.estimate_full_size / result["processed"] * result["api_calls"]) if result["processed"] > 0 else 0
    est_full_cost = est_full_calls * args.estimated_cost_per_call
    summary = {
        "mode": args.mode,
        "model": args.model,
        "batch_size": args.batch_size,
        "offset": args.offset,
        "input_rows": int(len(df if limit is None else df.iloc[:limit])),
        "processed": result["processed"],
        "api_calls": result["api_calls"],
        "final_groups": result["final_groups"],
        "elapsed_seconds": round(result["elapsed_seconds"], 2),
        "items_per_second": round(rate, 4),
        "eta_full_dataset_seconds": round(eta_seconds, 2) if math.isfinite(eta_seconds) else None,
        "eta_full_dataset_hms": time.strftime("%Hh %Mm %Ss", time.gmtime(eta_seconds)) if math.isfinite(eta_seconds) else None,
        "estimated_cost_per_call_usd": args.estimated_cost_per_call,
        "estimated_cost_usd": round(est_cost, 6),
        "estimated_full_api_calls": round(est_full_calls, 2),
        "estimated_full_cost_usd": round(est_full_cost, 6),
        "output_json": str(output_path),
    }

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2))
    print("\nBatching metrics:")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved groups  -> {groups_txt}")




if __name__ == "__main__":
    main()
