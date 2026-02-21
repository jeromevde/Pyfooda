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


# ---------------------------------------------------------------------------
# Search tool - token overlap + trigram Jaccard (no refit issues)
# ---------------------------------------------------------------------------


class FoodSearchIndex:
    """Lightweight search index over aggregated food names.

    Uses token overlap + character trigram Jaccard similarity.
    O(1) add, O(n) search - n stays small (target ~10-30k generic entries).
    """

    def __init__(self):
        self.names: list[str] = []
        self.ids: list[int] = []
        self._trigrams: list[set[str]] = []
        self._tokens: list[set[str]] = []

    def _tokenize(self, text: str):
        text = text.lower().strip()
        tokens = set(re.findall(r'\b\w+\b', text))
        padded = f' {text} '
        trigrams = {padded[i:i+3] for i in range(len(padded) - 2)}
        return tokens, trigrams

    def add(self, food_id: int, name: str):
        tokens, trigrams = self._tokenize(name)
        self.names.append(name.lower())
        self.ids.append(food_id)
        self._tokens.append(tokens)
        self._trigrams.append(trigrams)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top-k closest entries as [{id, name, score}]."""
        if not self.names:
            return []

        q_tokens, q_trigrams = self._tokenize(query)
        scores = []

        for i in range(len(self.names)):
            # Token Jaccard
            t_inter = len(q_tokens & self._tokens[i])
            t_union = len(q_tokens | self._tokens[i])
            token_score = t_inter / t_union if t_union > 0 else 0.0

            # Trigram Jaccard
            g_inter = len(q_trigrams & self._trigrams[i])
            g_union = len(q_trigrams | self._trigrams[i])
            trigram_score = g_inter / g_union if g_union > 0 else 0.0

            score = 0.4 * token_score + 0.6 * trigram_score
            scores.append(score)

        top_idx = np.argsort(scores)[-top_k:][::-1]
        results = []
        for i in top_idx:
            if scores[i] > 0.05:
                results.append({
                    'id': self.ids[i],
                    'name': self.names[i],
                    'score': round(float(scores[i]), 3),
                })
        return results

    def __len__(self):
        return len(self.names)


# ---------------------------------------------------------------------------
# Nutrient helpers
# ---------------------------------------------------------------------------

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


def _build_batch_prompt(foods: list[dict]) -> str:
    """Build a single user message containing a batch of foods to classify."""
    lines = []
    for f in foods:
        sr = f['search_results']
        if not sr:
            matches_str = '  (no existing entries yet)'
        else:
            matches_str = '\n'.join(
                f'    id={m["id"]}  "{m["name"]}"  [{m.get("nutrients", "")}]  (score={m["score"]})'
                for m in sr
            )
        nutrients_str = f.get('nutrients_str', '')
        lines.append(
            f'[{f["idx"]}] "{f["name"]}"  category={f["category"]}  [{nutrients_str}]\n'
            f'  Closest existing entries:\n{matches_str}'
        )
    return '\n\n'.join(lines)


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------


def _call_llm_batch(
    system_prompt: str,
    user_message: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.05,
    max_retries: int = 3,
) -> str:
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
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


def _parse_llm_response(raw: str, batch_size: int) -> list[dict]:
    """Parse LLM response lines.

    Expected format per line:
      [<idx>] CREATE <generic_name>
      [<idx>] ADD <id>
      [<idx>] IGNORE
    """
    # Try JSON first
    try:
        cleaned = raw
        if cleaned.startswith('```'):
            cleaned = '\n'.join(cleaned.split('\n')[1:])
        if cleaned.endswith('```'):
            cleaned = '\n'.join(cleaned.split('\n')[:-1])
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Line-by-line parsing with regex — flexible patterns
    results = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        # Standard: [42] CREATE Apple
        m = re.match(r'\[?(\d+)\]?\s*(CREATE|ADD|IGNORE)\s*(.*)', line, re.IGNORECASE)
        if not m:
            continue
        idx = int(m.group(1))
        action = m.group(2).upper()
        rest = m.group(3).strip()

        if action == 'CREATE':
            name = rest.strip().strip('"').strip("'").strip(',').strip()
            if name:
                results.append({'idx': idx, 'action': 'CREATE', 'name': name})
        elif action == 'ADD':
            rest_clean = rest.strip().strip('"').strip("'")
            # Try numeric ID first
            try:
                target_id = int(rest_clean.split()[0])
                results.append({'idx': idx, 'action': 'ADD', 'target_id': target_id})
            except (ValueError, IndexError):
                # Model gave a name instead of ID — resolve later
                if rest_clean:
                    results.append({'idx': idx, 'action': 'ADD', 'target_name': rest_clean})
        elif action == 'IGNORE':
            results.append({'idx': idx, 'action': 'IGNORE'})
    return results


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
        search_top_k: int = 5,
        checkpoint_dir: str = './checkpoints',
    ):
        self.source_df = df.copy().reset_index(drop=True)
        self.model = model
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY', '')
        self.base_url = base_url or os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        if not self.api_key:
            raise ValueError(
                'No API key found. Set OPENROUTER_API_KEY:\n'
                "  export OPENROUTER_API_KEY='sk-or-...'"
            )
        self.system_prompt = load_prompt(prompt_path)
        self.batch_size = batch_size
        self.search_top_k = search_top_k
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Aggregated database  {id -> entry}
        self.db: dict[int, dict] = {}
        self._next_id = 1
        self.index = FoodSearchIndex()

        # Stats
        self.stats = {'created': 0, 'added': 0, 'ignored': 0, 'errors': 0, 'api_calls': 0}
        self.processed_count = 0

    # -- public -----------------------------------------------------------------

    def run(self, limit: int | None = None, resume_from: int = 0):
        """Run the aggregation. limit=None processes all foods."""
        df = self.source_df
        # Filter out rows with missing food names
        df = df[df['foodName'].notna() & (df['foodName'].astype(str) != 'nan')].reset_index(drop=True)
        # Sort by category so similar foods land in the same batch
        df = df.sort_values('food_category', na_position='last').reset_index(drop=True)
        if limit:
            df = df.iloc[:limit]

        total = len(df)
        pbar = tqdm(total=total, initial=resume_from, desc='Aggregating')

        for batch_start in range(resume_from, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch_rows = df.iloc[batch_start:batch_end]

            # 1. build search results for each item
            foods = []
            for i, (_, row) in enumerate(batch_rows.iterrows()):
                name = str(row.get('foodName', ''))
                cat = str(row.get('food_category', ''))
                sr = self.index.search(name, top_k=self.search_top_k)
                # Enrich search results with nutrient fingerprints
                for m in sr:
                    entry = self.db.get(m['id'])
                    if entry:
                        m['nutrients'] = _nutrient_fingerprint(entry['nutrients'], is_dict=True)
                foods.append({
                    'idx': batch_start + i,
                    'name': name,
                    'category': cat,
                    'nutrients_str': _nutrient_fingerprint(row),
                    'search_results': sr,
                })

            # 2. call LLM
            user_msg = _build_batch_prompt(foods)
            try:
                raw = _call_llm_batch(
                    self.system_prompt,
                    user_msg,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.model,
                )
                self.stats['api_calls'] += 1
            except Exception as e:
                print(f'\n  LLM call failed for batch {batch_start}: {e}')
                self.stats['errors'] += len(foods)
                pbar.update(len(foods))
                continue

            # 3. parse & apply — with intra-batch dedup
            decisions = _parse_llm_response(raw, len(foods))
            decisions_by_idx = {d['idx']: d for d in decisions}

            # Track CREATEs within this batch so duplicates auto-merge
            batch_created: dict[str, int] = {}   # generic_name_lower -> db id

            for f in foods:
                row = df.iloc[f['idx']]
                decision = decisions_by_idx.get(f['idx'])

                if decision is None:
                    # parse miss — IGNORE unparsed items rather than creating
                    # catch-all entries from category names
                    self.stats['errors'] += 1
                    self.stats['ignored'] += 1
                elif decision['action'] == 'CREATE':
                    name = decision.get('name', f['name'])
                    name_key = name.strip().lower()
                    # Check if same generic name was already created in this batch
                    if name_key in batch_created:
                        self._do_add(batch_created[name_key], row)
                        self.stats['added'] += 1
                    else:
                        # Also check if search index has a very close match
                        existing = self.index.search(name, top_k=1)
                        if existing and existing[0]['score'] > 0.85:
                            self._do_add(existing[0]['id'], row)
                            self.stats['added'] += 1
                        else:
                            self._do_create(name, row)
                            batch_created[name_key] = self._next_id - 1
                            self.stats['created'] += 1
                elif decision['action'] == 'ADD':
                    target = decision.get('target_id')
                    target_name = decision.get('target_name')
                    # Resolve name-based ADD by searching the index
                    if not target and target_name:
                        matches = self.index.search(target_name, top_k=1)
                        if matches and matches[0]['score'] > 0.5:
                            target = matches[0]['id']
                        else:
                            # Also check batch_created for intra-batch names
                            key = target_name.strip().lower()
                            target = batch_created.get(key)
                    if target and target in self.db:
                        self._do_add(target, row)
                        self.stats['added'] += 1
                    else:
                        # Couldn't resolve — treat as CREATE with the target_name
                        create_name = target_name or f['name']
                        name_key = create_name.strip().lower()
                        if name_key in batch_created:
                            self._do_add(batch_created[name_key], row)
                            self.stats['added'] += 1
                        else:
                            existing = self.index.search(create_name, top_k=1)
                            if existing and existing[0]['score'] > 0.85:
                                self._do_add(existing[0]['id'], row)
                                self.stats['added'] += 1
                            else:
                                self._do_create(create_name, row)
                                batch_created[name_key] = self._next_id - 1
                                self.stats['created'] += 1
                elif decision['action'] == 'IGNORE':
                    self.stats['ignored'] += 1

                self.processed_count += 1

            pbar.update(len(foods))

            # Checkpoint every 500 items
            if self.processed_count % 500 < self.batch_size:
                self._save_checkpoint()

        pbar.close()
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

        self.db[fid] = {
            'id': fid,
            'generic_name': generic_name,
            'food_category': str(row.get('food_category', '')),
            'nutrients': nutrients,
            'source_ids': [int(row.name) if isinstance(row.name, (int, np.integer)) else 0],
            'source_names': [str(row.get('foodName', ''))],
            'portion_gram_weights': portion_weights,
            'portion_unit_names': portion_units,
            'count': 1,
        }
        self.index.add(fid, generic_name)

    def _do_add(self, target_id: int, row: pd.Series):
        entry = self.db[target_id]
        entry['nutrients'] = _avg_nutrients(entry['nutrients'], row, entry['count'])
        entry['count'] += 1
        src_id = int(row.name) if isinstance(row.name, (int, np.integer)) else 0
        entry['source_ids'].append(src_id)
        entry['source_names'].append(str(row.get('foodName', '')))
        # Collect serving size
        pgw = row.get('portion_gram_weight', np.nan)
        if pd.notna(pgw):
            entry['portion_gram_weights'].append(float(pgw))
        pun = row.get('portion_unit_name', '')
        if pd.notna(pun) and str(pun).strip():
            entry['portion_unit_names'].append(str(pun))

    # -- persistence ------------------------------------------------------------

    def _save_checkpoint(self):
        path = self.checkpoint_dir / 'aggregation_checkpoint.json'
        data = {
            'processed_count': self.processed_count,
            'next_id': self._next_id,
            'stats': self.stats,
            'db': self.db,
        }
        with open(path, 'w') as f:
            json.dump(data, f, default=str)

    def load_checkpoint(self) -> bool:
        path = self.checkpoint_dir / 'aggregation_checkpoint.json'
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        self.processed_count = data['processed_count']
        self._next_id = data['next_id']
        self.stats = data['stats']
        self.db = {int(k): v for k, v in data['db'].items()}
        self.index = FoodSearchIndex()
        for fid, entry in self.db.items():
            self.index.add(fid, entry['generic_name'])
        print(
            f'  Resumed from checkpoint: {self.processed_count} items processed, '
            f'{len(self.db)} generic foods'
        )
        return True

    # -- output -----------------------------------------------------------------

    def save(self, output_path: str = 'pyfooda/data/foods_aggregated.json'):
        """Save the compact aggregated database."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, 'w') as f:
            json.dump(list(self.db.values()), f, indent=2, default=str)
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
        print(f'  Parse errors (-> create): {self.stats["errors"]}')
        print(f'  API calls              : {self.stats["api_calls"]}')
        print(f'  Final DB size          : {len(self.db)} entries')
        print('=' * 60)
