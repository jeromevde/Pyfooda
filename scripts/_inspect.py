"""Quick inspection of aggregation results by category."""
import json, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else 'tests/test_aggregated.json'
with open(path) as f:
    data = json.load(f)

CATEGORIES = {
    'LENTILS': ['lentil'],
    'HAM': ['ham'],
    'APPLE PIE': ['apple', 'pie'],
    'YOGURT': ['yogurt', 'yoghurt', 'ygrt'],
    'LEMON': ['lemon', 'lemonade'],
}

def categorise(entry):
    name = entry['generic_name'].lower()
    sources = ' '.join(s.lower() for s in entry.get('source_names', []))
    text = name + ' ' + sources
    for cat, kws in CATEGORIES.items():
        if all(kw in text for kw in kws):
            return cat
    for cat, kws in CATEGORIES.items():
        if any(kw in text for kw in kws):
            return cat
    return 'OTHER'

groups = defaultdict(list)
for e in data:
    groups[categorise(e)].append(e)

total_sources = sum(e['count'] for e in data)
print(f"Total: {len(data)} groups, {total_sources} sources classified out of 244\n")

for cat in list(CATEGORIES.keys()) + ['OTHER']:
    entries = groups.get(cat, [])
    if not entries:
        continue
    cat_sources = sum(e['count'] for e in entries)
    print(f"  {cat} ({len(entries)} groups, {cat_sources} sources):")
    for e in sorted(entries, key=lambda x: -x['count']):
        energy = e['nutrients'].get('Energy')
        e_str = f"E={energy:.0f}" if energy else ""
        names = e.get('source_names', [])
        shown = ', '.join(names[:3])
        more = f" +{len(names)-3} more" if len(names) > 3 else ""
        print(f"    {e['generic_name']:40s} count={e['count']:2d}  {e_str}")
        print(f"      sources: {shown}{more}")
    print()
