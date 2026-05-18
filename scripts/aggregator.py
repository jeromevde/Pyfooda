from __future__ import annotations

"""
Food Aggregation Engine - Tool-based approach

Instead of computing a full similarity matrix, we process foods one-by-one
(in LLM batches) and let the LLM decide:

  CREATE <generic_name>  - start a new generic food entry
  ADD <id>               - merge into an existing generic entry
  IGNORE                 - skip this food (too niche / unclear)

A lightweight search tool gives the LLM the top-k closest entries already
in the aggregated DB so it can make an informed decision.

The aggregated DB is kept compact:
  {id, generic_name, food_category, nutrients, source_ids[], count}
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import openai

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


# Energy-based nutrient guard: reject additions that are too far from the group
# Uses MAD (Median Absolute Deviation) — robust even under 50% contamination.
# Falls back to a fixed ±150 kcal band when MAD is near zero (uniform groups).

_ENERGY_MAD_THRESHOLD = 3.5   # multiples of MAD
_ENERGY_FIXED_BAND = 200      # kcal, fallback when MAD < 10
_ENERGY_MIN_COUNT = 3         # don't gate until we have this many sources
_ENERGY_EXTREME_RATIO = 10.0  # always reject if new/group ratio exceeds this (fires even with 1 sample)


def _is_energy_compatible(entry: dict, new_energy: float) -> bool:
    """Return True if new_energy is compatible with the group's energy profile.

    Two-stage check:
    1. Extreme-ratio guard (fires with ≥1 sample): rejects when new_energy is
       more than _ENERGY_EXTREME_RATIO times the group median.  Catches e.g.
       broth (16 kcal) being added to a dry-soup group (400+ kcal) regardless
       of how many sources the group already has.
    2. MAD/band gate (fires at ≥_ENERGY_MIN_COUNT samples): rejects outliers
       beyond max(3.5*MAD, 200 kcal) from the group median.
    """
    energies = entry.get('_source_energies', [])
    if not energies:
        return True

    arr = np.array(energies, dtype=float)
    median = float(np.median(arr))

    # Stage 1: extreme ratio guard — always active
    if median > 0 and new_energy > 0:
        ratio = max(median, new_energy) / min(median, new_energy)
        if ratio > _ENERGY_EXTREME_RATIO:
            return False

    # Stage 2: MAD/band gate — needs enough samples to be meaningful
    if len(energies) < _ENERGY_MIN_COUNT:
        return True

    mad = float(np.median(np.abs(arr - median)))
    # Use the more lenient threshold — the fixed band provides a floor
    # so that groups with low MAD don't become overly restrictive
    threshold = max(_ENERGY_MAD_THRESHOLD * mad, _ENERGY_FIXED_BAND)
    return abs(new_energy - median) <= threshold


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
    print(
        '  OPENROUTER_API_KEY not set — falling back to VS Code Copilot proxy '
        f'at {COPILOT_PROXY_BASE_URL}  (model: {COPILOT_PROXY_MODEL})'
    )
    return 'dummy', COPILOT_PROXY_BASE_URL, COPILOT_PROXY_MODEL


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
      [idx] RENAME <id> <new_name_existing> CREATE <new_name_incoming>
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
        if re.match(r'^(CREATE|ADD|RENAME|IGNORE)\b', line, re.IGNORECASE):
            line = f'[{expected_idx}] ' + line

        rm = re.match(
            r'\[?(\d+)\]?\s*RENAME\s+(\d+)\s+["\']?([^"\']+?)["\']?'
            r'(?:\s+(?:CREATE|->)\s+["\']?([^"\']+?)["\']?)?\s*$',
            line, re.IGNORECASE
        )
        if rm:
            return {
                'idx': expected_idx,
                'action': 'RENAME',
                'target_id': int(rm.group(2)),
                'new_name_existing': rm.group(3).strip(),
                'new_name_incoming': rm.group(4).strip() if rm.group(4) else None,
            }

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
        agg.run(limit=1000)            # test run
        agg.save('output.json')
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        model: str = 'google/gemini-2.0-flash-lite-001',
        api_key: str | None = None,
        base_url: str | None = None,
        prompt_path: str | None = None,
        batch_size: int = 50,
        search_top_k: int = 8,
        checkpoint_dir: str = './checkpoints',
        timeout_seconds: int = 600,
    ):
        self.source_df = df.copy().reset_index(drop=True)
        self.api_key, self.base_url, self.model = _resolve_llm_backend(api_key, base_url, model)
        self.system_prompt = load_prompt(prompt_path)
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

    # -- public -----------------------------------------------------------------

    def run(self, limit: int | None = None, resume_from: int = 0):
        """Run streaming aggregation (one food per LLM call)."""
        df = self.source_df
        df = df[df['foodName'].notna() & (df['foodName'].astype(str) != 'nan')].reset_index(drop=True)

        # Foundation/legacy first; branded later for cleaner base groups.
        df['_dtype_rank'] = df['data_type'].apply(_data_type_rank)
        df = df.sort_values(
            ['_dtype_rank', 'food_category'],
            ascending=[True, True],
            na_position='last',
        ).reset_index(drop=True)
        if limit:
            df = df.iloc[:limit]

        total = len(df)
        pbar = tqdm(total=total, initial=resume_from, desc='Aggregating (streaming)')

        for idx in range(resume_from, total):
            row = df.iloc[idx]
            name = str(row.get('foodName', ''))
            cat = str(row.get('food_category', ''))

            search_results = self.index.search(name, top_k=self.search_top_k)
            for m in search_results:
                entry = self.db.get(m['id'])
                if entry:
                    m['nutrients'] = _nutrient_fingerprint(entry['nutrients'], is_dict=True)
                    m['count'] = entry.get('count', 1)

            food = {
                'idx': idx,
                'name': name,
                'category': cat,
                'nutrients_str': _nutrient_fingerprint(row),
                'search_results': search_results,
            }

            user_msg = _build_item_prompt(food)
            try:
                raw = _call_llm_batch(
                    self.system_prompt,
                    user_msg,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.model,
                    timeout_seconds=self.timeout_seconds,
                )
                self.stats['api_calls'] += 1
            except Exception as e:
                print(f'\n  LLM call failed for idx {idx}: {e}')
                self.stats['errors'] += 1
                self.stats['ignored'] += 1
                self.processed_count += 1
                pbar.update(1)
                continue

            decision = _parse_llm_decision(raw, idx)
            if decision is None:
                self.stats['errors'] += 1
                self.stats['ignored'] += 1
                self.processed_count += 1
                pbar.update(1)
                continue

            if decision['action'] == 'CREATE':
                create_name = decision.get('name', name)
                name_key = create_name.strip().lower()
                if name_key in self._name_to_id:
                    existing_id = self._name_to_id[name_key]
                    result = self._do_add(existing_id, row)
                    if result == 'ok':
                        self.stats['added'] += 1
                        self.stats.setdefault('name_dedup_merged', 0)
                        self.stats['name_dedup_merged'] += 1
                    else:
                        self._do_create(create_name, row)
                        self.stats['created'] += 1
                elif (existing := self.index.search(create_name, top_k=1)) and existing[0]['score'] > 0.88:
                    result = self._do_add(existing[0]['id'], row)
                    if result == 'ok':
                        self.stats['added'] += 1
                    else:
                        self._do_create(create_name, row)
                        self.stats['created'] += 1
                else:
                    self._do_create(create_name, row)
                    self.stats['created'] += 1

            elif decision['action'] == 'ADD':
                target = decision.get('target_id')
                target_name = decision.get('target_name')
                valid_add_ids = food.get('_valid_add_ids', set())

                if target is not None and target not in valid_add_ids:
                    self.stats.setdefault('add_id_rejected', 0)
                    self.stats['add_id_rejected'] += 1
                    target = None

                if target is None and target_name:
                    matches = self.index.search(target_name, top_k=1)
                    if matches and matches[0]['score'] > 0.75 and matches[0]['id'] in valid_add_ids:
                        target = matches[0]['id']

                if target and target in self.db:
                    result = self._do_add(target, row)
                    if result == 'ok':
                        self.stats['added'] += 1
                    elif result == 'foundation_locked':
                        self.stats['ignored'] += 1
                    else:
                        create_name = target_name or name
                        self._do_create(create_name, row)
                        self.stats['created'] += 1
                else:
                    create_name = target_name or name
                    self._do_create(create_name, row)
                    self.stats['created'] += 1

            elif decision['action'] == 'RENAME':
                target_id = decision.get('target_id')
                new_name_existing = decision.get('new_name_existing', '').strip()
                new_name_incoming = decision.get('new_name_incoming')
                if target_id and target_id in self.db and new_name_existing:
                    old_name_key = self.db[target_id]['generic_name'].strip().lower()
                    self._name_to_id.pop(old_name_key, None)
                    self.db[target_id]['generic_name'] = new_name_existing
                    self._name_to_id[new_name_existing.lower()] = target_id
                    self.index.rename(target_id, new_name_existing)
                    self.stats.setdefault('renamed', 0)
                    self.stats['renamed'] += 1

                    create_name = (new_name_incoming or name).strip()
                    if create_name.lower() in self._name_to_id:
                        result = self._do_add(self._name_to_id[create_name.lower()], row)
                        if result == 'ok':
                            self.stats['added'] += 1
                        else:
                            self._do_create(create_name, row)
                            self.stats['created'] += 1
                    else:
                        self._do_create(create_name, row)
                        self.stats['created'] += 1
                else:
                    self.stats['errors'] += 1
                    self.stats['ignored'] += 1

            elif decision['action'] == 'IGNORE':
                self.stats['ignored'] += 1
            else:
                self.stats['errors'] += 1
                self.stats['ignored'] += 1

            self.processed_count += 1
            pbar.update(1)

            if self.processed_count % 500 == 0:
                self._save_checkpoint()

        pbar.close()
        self._apply_similarity_merge_postpass(threshold=0.80)
        self._apply_canonical_postpass()
        self._save_checkpoint()
        self._print_summary()

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
        """Add a source food to an existing group.

        Returns:
            'ok'                  – successfully added
            'energy_rejected'     – energy outlier gate fired
            'foundation_locked'   – group is foundation-locked, lower source rejected

        Foundation-locked groups:
        If a group was created by a foundation_food, it already has
        lab-measured, gold-standard nutrients.  Lower-quality sources
        (legacy / branded) are rejected to keep the group clean.
        """
        entry = self.db[target_id]

        # Foundation lock — protect lab-quality groups from dilution
        created_by = entry.get('_created_by', '')
        incoming_dtype = str(row.get('data_type', ''))
        if created_by == 'foundation_food' and incoming_dtype != 'foundation_food':
            self.stats['foundation_locked'] += 1
            return 'foundation_locked'

        # Energy-based outlier gate
        new_energy = row.get('Energy', np.nan)
        if pd.notna(new_energy) and not _is_energy_compatible(entry, float(new_energy)):
            return 'energy_rejected'

        entry['nutrients'] = _avg_nutrients(entry['nutrients'], row, entry['count'])
        entry['count'] += 1
        src_id = int(row.name) if isinstance(row.name, (int, np.integer)) else 0
        entry['source_ids'].append(src_id)
        entry['source_names'].append(str(row.get('foodName', '')))
        # Track energy for future gating
        if pd.notna(new_energy):
            entry.setdefault('_source_energies', []).append(float(new_energy))
        # Collect serving size
        pgw = row.get('portion_gram_weight', np.nan)
        if pd.notna(pgw):
            entry['portion_gram_weights'].append(float(pgw))
        pun = row.get('portion_unit_name', '')
        if pd.notna(pun) and str(pun).strip():
            entry['portion_unit_names'].append(str(pun))

        return 'ok'


    def _canonical_group_name(self, entry: dict) -> str | None:
        """Return a generic normalization key for safe post-pass dedup only.

        Important: no dataset-specific hardcoded food families here.
        """
        raw = str(entry.get('generic_name', '')).strip()
        if not raw:
            return None

        name = raw.lower()
        # conservative normalization: punctuation/spacing/case only
        name = re.sub(r'[^a-z0-9\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        if not name:
            return None
        return name.title()

    def _merge_entries(self, base: dict, incoming: dict):
        """Merge incoming aggregated entry into base (weighted nutrients)."""
        base_count = int(base.get('count', 1))
        inc_count = int(incoming.get('count', 1))

        for col in NUTRIENT_COLS:
            a = base.get('nutrients', {}).get(col)
            b = incoming.get('nutrients', {}).get(col)
            if a is None and b is None:
                v = None
            elif a is None:
                v = b
            elif b is None:
                v = a
            else:
                v = (float(a) * base_count + float(b) * inc_count) / (base_count + inc_count)
            base['nutrients'][col] = v

        base['count'] = base_count + inc_count
        base.setdefault('source_ids', []).extend(incoming.get('source_ids', []))
        base.setdefault('source_names', []).extend(incoming.get('source_names', []))
        base.setdefault('portion_gram_weights', []).extend(incoming.get('portion_gram_weights', []))
        base.setdefault('portion_unit_names', []).extend(incoming.get('portion_unit_names', []))
        base.setdefault('_source_energies', []).extend(incoming.get('_source_energies', []))

    def _apply_similarity_merge_postpass(self, threshold: float = 0.80):
        """Merge same-category groups with high embedding similarity and compatible cooking states."""
        if not self.db:
            return

        COOKED_STATES = {'cooked', 'baked', 'fried', 'smoked', 'roasted', 'broiled',
                         'boiled', 'grilled', 'stewed', 'steamed', 'poached', 'braised'}
        PREP_CONFLICT = [
            # raw/fresh vs any cooked → conflict
            ({'raw', 'fresh'}, COOKED_STATES),
            # frozen vs fresh/raw → conflict
            ({'frozen'}, {'raw', 'fresh'}),
            # frozen vs non-frozen cooked → conflict (frozen-cooked ≠ fresh-cooked)
            ({'frozen'}, COOKED_STATES),
            # canned vs raw/fresh/frozen → conflict
            ({'canned'}, {'raw', 'fresh', 'frozen'}),
            # dried vs raw/fresh/cooked → conflict
            ({'dried'}, {'raw', 'fresh', 'cooked'}),
            # smoked vs raw/fresh → conflict
            ({'smoked'}, {'raw', 'fresh'}),
            # oil-roasted vs dry-roasted → different fat method
            ({'oil'}, {'dry'}),
            # fried vs baked/broiled/roasted (different fat level)
            ({'fried'}, {'baked', 'broiled', 'roasted'}),
        ]

        # Stop-words for food-token jaccard check
        STOP = {'raw', 'cooked', 'baked', 'fried', 'smoked', 'dried', 'roasted', 'broiled',
                'boiled', 'grilled', 'stewed', 'steamed', 'with', 'without', 'oil', 'fat',
                'added', 'no', 'free', 'plain', 'unsalted', 'whole', 'mixed', 'species',
                'regular', 'lightly', 'reduced', 'low', 'high', 'light', 'dark', 'mild',
                'heat', 'dry', 'moist', 'and', 'or', 'the', 'of', 'from', 'to', 'in', 'on',
                'at', 'nfs', 'ns', 'all', 'any', 'type', 'form', 'meat', 'only', 'skin',
                'eaten', 'not', 'as', 'fresh', 'frozen', 'canned', 'n', 's', 'a'}

        def food_tokens(name: str) -> set:
            words = set(re.findall(r'\b[a-z]{2,}\b', name.lower()))
            return words - STOP

        def entry_words(entry: dict) -> set:
            """Collect all significant prep/cooking words from generic_name + source_names."""
            words: set = set()
            words.update(re.findall(r'\b\w+\b', entry.get('generic_name', '').lower()))
            for sn in entry.get('source_names', [])[:6]:
                words.update(re.findall(r'\b\w+\b', sn.lower()))
            return words

        def _fat_tier_label(name: str) -> str:
            """Classify a group name into a fat tier.

            Returns one of: 'fat_free', 'low_fat', 'reduced', 'light', 'regular'.
            Uses word-boundary matching to avoid false positives (e.g. 'lightly'
            must NOT match 'light').
            """
            n = name.lower()
            if re.search(r'\b(fat.?free|nonfat)\b', n):
                return 'fat_free'
            if re.search(r'\blow.?fat\b', n):
                return 'low_fat'
            if re.search(r'\breduced.?fat\b', n):
                return 'reduced'
            if re.search(r'\b(light|lite|part.?skim)\b', n):
                return 'light'
            return 'regular'

        def conflict(entry_a: dict, entry_b: dict) -> bool:
            # Use full word sets (generic_name + source_names) so e.g. 'raw' from
            # "Chicken, stewing, dark meat, meat only, raw" is visible even when
            # the generic name is "Chicken Stewing Dark Meat Meat Only"
            w1 = entry_words(entry_a)
            w2 = entry_words(entry_b)
            for a_set, b_set in PREP_CONFLICT:
                if (w1 & a_set) and (w2 & b_set):
                    return True
                if (w2 & a_set) and (w1 & b_set):
                    return True
            # Generalized fat-tier conflict: any two groups in different tiers
            # must not merge (fat-free ≠ low-fat ≠ light ≠ reduced ≠ regular).
            n1 = entry_a.get('generic_name', '').lower()
            n2 = entry_b.get('generic_name', '').lower()
            t1, t2 = _fat_tier_label(n1), _fat_tier_label(n2)
            if t1 != t2:
                return True
            return False

        def food_jaccard(n1: str, n2: str) -> float:
            t1 = food_tokens(n1)
            t2 = food_tokens(n2)
            if not t1 or not t2:
                return 0.0
            return len(t1 & t2) / len(t1 | t2)

        # Union-Find for clustering
        parent = {gid: gid for gid in self.db}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int):
            px, py = find(x), find(y)
            if px != py:
                if self.db.get(px, {}).get('count', 0) >= self.db.get(py, {}).get('count', 0):
                    parent[py] = px
                else:
                    parent[px] = py

        gids = list(self.db.keys())
        for gid in gids:
            entry = self.db.get(gid)
            if entry is None:
                continue
            name = entry['generic_name']
            cat = entry.get('food_category', '')
            results = self.index.search(name, top_k=10)
            for r in results:
                if r['id'] == gid or r['score'] < threshold:
                    continue
                other = self.db.get(r['id'])
                if other is None:
                    continue
                if other.get('food_category', '') != cat:
                    continue
                if conflict(entry, other):
                    continue
                # Require at least one shared food token (prevents e.g. asparagus + broccoli)
                if food_jaccard(name, other['generic_name']) < 0.2:
                    continue
                # Nutrient gate: block merge if energy differs >40% or fat ratio differs >60%
                e_a = entry.get('nutrients', {}).get('Energy')
                e_b = other.get('nutrients', {}).get('Energy')
                if e_a and e_b and e_a > 0 and e_b > 0:
                    if abs(e_a - e_b) / min(e_a, e_b) > 0.40:
                        continue
                f_a = entry.get('nutrients', {}).get('Total fat')
                f_b = other.get('nutrients', {}).get('Total fat')
                if f_a is not None and f_b is not None:
                    max_fat = max(f_a, f_b)
                    if max_fat > 1.0 and abs(f_a - f_b) / max_fat > 0.60:
                        continue
                union(gid, r['id'])

        # Apply merges
        root_to_members: dict[int, list[int]] = {}
        for gid in gids:
            root = find(gid)
            root_to_members.setdefault(root, []).append(gid)

        merges = 0
        for root, members in root_to_members.items():
            if len(members) <= 1:
                continue
            base_id = max(members, key=lambda g: self.db.get(g, {}).get('count', 0))
            for mid in members:
                if mid == base_id:
                    continue
                if mid in self.db and base_id in self.db:
                    self._merge_entries(self.db[base_id], self.db[mid])
                    del self.db[mid]
                    merges += 1

        self.stats['similarity_merged_groups'] = merges
        if merges:
            self._next_id = max(self.db.keys(), default=0) + 1
            self.index = FoodSearchIndex()
            self._name_to_id = {}
            for fid, e in self.db.items():
                self.index.add(fid, e['generic_name'])
                self._name_to_id[e['generic_name'].strip().lower()] = fid

    def _apply_canonical_postpass(self):
        """Deterministic cleanup pass to collapse known semantic variants."""
        if not self.db:
            return

        new_db = {}
        canonical_to_id = {}
        next_id = 1

        for _, entry in sorted(self.db.items(), key=lambda kv: kv[0]):
            target_name = self._canonical_group_name(entry) or entry.get('generic_name', '')
            entry_cat = entry.get('food_category', '')
            # Scope dedup key by category so groups from different categories never merge
            key = f"{entry_cat}||{target_name.strip().lower()}"

            if key not in canonical_to_id:
                eid = next_id
                next_id += 1
                clone = dict(entry)
                clone['id'] = eid
                clone['generic_name'] = target_name
                new_db[eid] = clone
                canonical_to_id[key] = eid
            else:
                base_id = canonical_to_id[key]
                self._merge_entries(new_db[base_id], entry)

        old_size = len(self.db)
        self.db = new_db
        self._next_id = max(self.db.keys(), default=0) + 1

        # Rebuild search/index maps for consistency
        self.index = FoodSearchIndex()
        self._name_to_id = {}
        for fid, e in self.db.items():
            self.index.add(fid, e['generic_name'])
            self._name_to_id[e['generic_name'].strip().lower()] = fid

        self.stats['postpass_merged_groups'] = old_size - len(self.db)

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
        print(f'  Renamed existing groups : {self.stats.get("renamed", 0)}')
        print(f'  Parse errors (-> create): {self.stats["errors"]}')
        print(f'  API calls              : {self.stats["api_calls"]}')
        print(f'  Final DB size          : {len(self.db)} entries')
        print(f'  Similarity merged groups: {self.stats.get("similarity_merged_groups", 0)}')
        print(f'  Post-pass merged groups: {self.stats.get("postpass_merged_groups", 0)}')
        print('=' * 60)
