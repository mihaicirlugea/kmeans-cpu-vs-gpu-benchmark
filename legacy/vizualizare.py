import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('date_proiect.csv')
plt.figure(figsize=(10, 6))
plt.scatter(df['X'], df['Y'], s=1, alpha=0.5)
plt.title("Distribuția celor 100.000 de instanțe")
plt.savefig('grafic_date.png') # Salvează poza automat pentru Word
plt.show()