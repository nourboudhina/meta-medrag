import os
import requests
from pathlib import Path
from getpass import getpass

# Tes credentials PhysioNet
username = input("PhysioNet username: ")
password = getpass("PhysioNet password: ")

session = requests.Session()
session.auth = (username, password)

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.0.0"

# Fichiers CSV à télécharger en premier (petits)
csv_files = [
    "mimic-cxr-2.0.0-split.csv.gz",
    "mimic-cxr-2.0.0-metadata.csv.gz", 
    "mimic-cxr-2.0.0-chexpert.csv.gz",
]

out_dir = Path("data/raw/mimic_cxr")
out_dir.mkdir(parents=True, exist_ok=True)

print("\nTelechargement des fichiers CSV...")
for fname in csv_files:
    url = f"{BASE_URL}/{fname}"
    out_path = out_dir / fname
    if out_path.exists():
        print(f"  Deja present: {fname}")
        continue
    print(f"  Telechargement: {fname}...")
    r = session.get(url, stream=True)
    if r.status_code == 200:
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  OK: {out_path}")
    else:
        print(f"  ERREUR {r.status_code}: {url}")

# Liste les patients dans p10 pour telecharger un sous-ensemble
print("\nRecuperation de la liste des patients p10...")
r = session.get(f"{BASE_URL}/files/p10/")
if r.status_code == 200:
    # Extrait les IDs patients depuis le HTML
    import re
    patients = re.findall(r'href="(p\d+)/"', r.text)[:20]  # 20 premiers patients
    print(f"  Patients trouves: {len(patients)}")
    
    img_dir = out_dir / "images" / "p10"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    for pat in patients[:5]:  # 5 patients pour le POC
        pat_url = f"{BASE_URL}/files/p10/{pat}/"
        r2 = session.get(pat_url)
        studies = re.findall(r'href="(s\d+)/"', r2.text)
        
        for study in studies[:2]:  # 2 études par patient
            study_url = f"{BASE_URL}/files/p10/{pat}/{study}/"
            r3 = session.get(study_url)
            images = re.findall(r'href="([^"]+\.jpg)"', r3.text)
            
            study_dir = img_dir / pat / study
            study_dir.mkdir(parents=True, exist_ok=True)
            
            for img_name in images[:2]:  # 2 images par étude
                img_url = f"{BASE_URL}/files/p10/{pat}/{study}/{img_name}"
                img_path = study_dir / img_name
                if not img_path.exists():
                    r4 = session.get(img_url)
                    if r4.status_code == 200:
                        img_path.write_bytes(r4.content)
                        print(f"    OK: {pat}/{study}/{img_name}")

print("\nDone ! Verifie data/raw/mimic_cxr/")