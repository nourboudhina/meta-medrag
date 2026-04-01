import json
from pathlib import Path

rep_dir = Path('data/raw/iu_xray/reports')
img_dir = Path('data/raw/iu_xray/images')

# Index toutes les images par leur numero CXR
# CXR1000_IM-0003-1001.png -> cle "1000"
img_index = {}
for img in img_dir.glob('**/*.png'):
    parts = img.name.replace('CXR', '').split('_')
    if parts:
        cxr_id = parts[0]
        if cxr_id not in img_index:
            img_index[cxr_id] = []
        img_index[cxr_id].append(img)

print(f'IDs images uniques: {len(img_index)}')
print(f'Exemple cles: {list(img_index.keys())[:5]}')

# Charge les rapports XML pour trouver le uID
import xml.etree.ElementTree as ET
xml_dir = Path('data/raw/iu_xray/reports_raw')
xml_files = list(xml_dir.glob('**/*.xml'))
print(f'Fichiers XML: {len(xml_files)}')

# Regarde la structure d un XML
if xml_files:
    tree = ET.parse(xml_files[0])
    root = tree.getroot()
    print(f'Tags racine: {[child.tag for child in root]}')
    # Cherche l ID du patient/image
    for elem in root.iter():
        if 'id' in elem.tag.lower() or elem.get('id'):
            print(f'  Tag: {elem.tag}, attribs: {elem.attrib}, text: {elem.text}')
            break