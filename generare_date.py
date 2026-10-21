import pandas as pd
from sklearn.datasets import make_blobs
import os

def genereaza_set_date(n_samples, filename):
    print(f"Generăm {n_samples:,} de instanțe...")
    X, _ = make_blobs(n_samples=n_samples, centers=4, n_features=2, random_state=42)
    
    df = pd.DataFrame(X, columns=['X', 'Y'])
    df.to_csv(filename, index=False)
    
    dimensiune = os.path.getsize(filename) / (1024 * 1024)
    print(f" Salvat: {filename} ({dimensiune:.2f} MB)")

# Generăm cele 3 praguri pentru testul de scalabilitate de la S12
genereaza_set_date(100000, 'date_100k.csv')
genereaza_set_date(1000000, 'date_1m.csv')
genereaza_set_date(5000000, 'date_5m.csv')