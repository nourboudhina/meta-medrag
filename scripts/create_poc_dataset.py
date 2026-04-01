import json, random
from pathlib import Path

random.seed(42)
data = json.load(open('data/raw/iu_xray/splits.json'))
data = [x for x in data if x.get('image')]

train = random.sample([x for x in data if x['split']=='train'], 100)
val   = random.sample([x for x in data if x['split']=='val'],   50)
test  = random.sample([x for x in data if x['split']=='test'],  50)

poc = train + val + test
Path('data/raw/iu_xray_poc').mkdir(parents=True, exist_ok=True)
json.dump(poc, open('data/raw/iu_xray_poc/splits.json','w'), indent=2)
print(f'POC dataset: {len(train)} train / {len(val)} val / {len(test)} test')