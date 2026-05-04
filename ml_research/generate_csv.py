import pandas as pd
import numpy as np
import os
import holidays
from datetime import datetime, timedelta

# Pad instellen
folder_path = './ml_research/data'
file_path = os.path.join(folder_path, 'data.csv')

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 1. Tijdsperiode instellen
start_date = datetime(2024, 1, 1)
end_date = datetime(2026, 12, 31) # We trekken het door tot eind 2026
dates = pd.date_range(start_date, end_date)

# 2. Belgische feestdagen ophalen
be_holidays = holidays.BE(years=[2024, 2025, 2026])

df = pd.DataFrame({'datum': dates})
df['dag_van_de_week'] = df['datum'].dt.weekday # Maandag = 0
df['maand'] = df['datum'].dt.month
df['is_weekend'] = df['dag_van_de_week'].apply(lambda x: 1 if x >= 5 else 0)

# 3. Check voor Belgische feestdagen
df['is_feestdag'] = df['datum'].apply(lambda x: 1 if x in be_holidays else 0)

# 4. Verkoop genereren
def calculate_sales(row):
    # Maandag gesloten (Maandag = 0 in pandas)
    if row['dag_van_de_week'] == 0:
        return 0
    
    # Basis verkoop
    base = 100
    # Weekend effect (Zaterdag/Zondag drukker)
    weekend = 60 if row['is_weekend'] == 1 else 0
    # Feestdag effect (Mensen kopen meer brood vooraf of op de dag zelf)
    holiday = 45 if row['is_feestdag'] == 1 else 0
    # Seizoenseffect
    season = np.sin(row['maand'] * (2 * np.pi / 12)) * 20
    # Willekeurige ruis
    noise = np.random.normal(0, 5)
    
    return max(0, int(base + weekend + holiday + season + noise))

df['verkoop_target'] = df.apply(calculate_sales, axis=1)

# Opslaan
df.to_csv(file_path, index=False)
print(f"✅ CSV aangemaakt met Belgische feestdagen en maandagssluiting.")