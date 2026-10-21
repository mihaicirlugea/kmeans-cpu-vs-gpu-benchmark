import pandas as pd
import time
import numpy as np
from sklearn.cluster import KMeans
from joblib import parallel_config

def warm_up():
    print("Încălzim siliciul (Warm-up)...")
    X_dummy = np.random.rand(10000, 2)
    KMeans(n_clusters=4, n_init='auto', random_state=42).fit(X_dummy)
    print(" Sistem stabilizat. Pornim cronometrele.\n")

def ruleaza_test(filename):
    print(f"==========================================")
    print(f" TEST PENTRU: {filename}")
    print(f"==========================================")
    
    df = pd.read_csv(filename)
    X = df.values

    # Secvențial
    model_seq = KMeans(n_clusters=4, n_init='auto', random_state=42)
    start_seq = time.time()
    model_seq.fit(X)
    timp_seq = time.time() - start_seq
    print(f" Timp Secvențial: {timp_seq:.4f} s")

    # Paralel
    model_par = KMeans(n_clusters=4, n_init='auto', random_state=42)
    start_par = time.time()
    with parallel_config(backend='loky', n_jobs=-1):
        model_par.fit(X)
    timp_par = time.time() - start_par
    print(f" Timp Paralel CPU: {timp_par:.4f} s")

    # Analiza
    procent = ((timp_par - timp_seq) / timp_seq) * 100
    if timp_par > timp_seq:
        print(f" Paralelizare ineficientă: Mai LENT cu {procent:.2f}%\n")
    else:
        print(f"Scalare reușită: Mai RAPID cu {abs(procent):.2f}%\n")

if __name__ == "__main__":
    warm_up()
    ruleaza_test('date_100k.csv')
    ruleaza_test('date_1m.csv')
    ruleaza_test('date_5m.csv')