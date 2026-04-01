import requests
import os
from pathlib import Path
from tqdm import tqdm

# Va sur physionet.org/settings/ et copie ton username
USERNAME = input("PhysioNet username: ")
PASSWORD = input("PhysioNet password: ")

# PhysioNet utilise une session avec login
session = requests.Session()

# Login via le formulaire web
login_url = "https://physionet.org/login/"
resp = session.get(login_url)

# Extrait le CSRF token
import re
csrf = re.search(r'csrfmiddlewaretoken.*?value="([^"]+)"', resp.text)
if csrf:
    csrf_token = csrf.group(1)
else:
    csrf_token = session.cookies.get('csrftoken', '')

# Login
login_data = {
    'username': USERNAME,
    'password': PASSWORD,
    'csrfmiddlewaretoken': csrf_token,
    'next': '/'
}
headers = {'Referer': login_url}
resp = session.post(login_url, data=login_data, headers=headers)

if 'login' in resp.url:
    print("ERREUR: Login echoue. Verifie username/password")
    exit(1)

print("Login OK!")

BASE = "https://physionet.org/files/mimic-cxr-jpg/2.0.0"
out = Path("data/raw/mimic_cxr")
out.mkdir(parents=True, exist_ok=True)

# Telecharge les CSVs
csvs = [
    "mimic-cxr-2.0.0-split.csv.gz",
    "mimic-cxr-2.0.0-metadata.csv.gz",
    "mimic-cxr-2.0.0-chexpert.csv.gz",
]

for fname in csvs:
    fpath = out / fname
    if fpath.exists():
        print(f"Deja present: {fname}")
        continue
    url = f"{BASE}/{fname}"
    print(f"Telechargement {fname}...")
    r = session.get(url, stream=True)
    if r.status_code == 200:
        total = int(r.headers.get('content-length', 0))
        with open(fpath, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"  OK: {fpath}")
    else:
        print(f"  ERREUR {r.status_code}: {url}")

# Telecharge 5 patients de p10 pour le POC
print("\nTelechargement images patients p10...")
img_out = out / "images"

for patient_num in range(10000032, 10000037):  # 5 patients
    pat_id = f"p{patient_num}"
    pat_url = f"{BASE}/files/p10/{pat_id}/"
    r = session.get(pat_url)
    if r.status_code != 200:
        continue
    
    studies = re.findall(r'href="(s\d+)/"', r.text)
    for study in studies[:1]:
        study_url = f"{BASE}/files/p10/{pat_id}/{study}/"
        r2 = session.get(study_url)
        images = re.findall(r'href="([a-f0-9\-]+\.jpg)"', r2.text)
        
        study_dir = img_out / "p10" / pat_id / study
        study_dir.mkdir(parents=True, exist_ok=True)
        
        for img_name in images[:2]:
            img_path = study_dir / img_name
            if img_path.exists():
                continue
            img_url = f"{BASE}/files/p10/{pat_id}/{study}/{img_name}"
            r3 = session.get(img_url)
            if r3.status_code == 200:
                img_path.write_bytes(r3.content)
                print(f"  OK: {pat_id}/{study}/{img_name} ({len(r3.content)/1024:.0f} KB)")

print("\nDone! Verifie data/raw/mimic_cxr/")