import sys 
from pathlib import Path 
Path('scripts').mkdir(exist_ok=True) 
code = open('scripts/_template.py', 'r', errors='ignore').read() if Path('scripts/_template.py').exists() else '' 
