import json
from pathlib import Path

rep_dir = Path('data/raw/iu_xray/reports')
img_dir = Path('data/raw/iu_xray/images')

reps = sorted(rep_dir.glob('*.txt'))
print(f'Rapports: {len(reps)}')

imgs = list(img_dir.glob('**/*.png'))
print(f'Images: {len(imgs)}')

if imgs:
    print(f'Exemple image: {imgs[0].name}')
if reps:
    print(f'Exemple rapport: {reps[0].name}')