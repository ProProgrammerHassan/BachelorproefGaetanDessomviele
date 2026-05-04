import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# 1. Geavanceerde Mock Data (Tijdreeks van 2 jaar)
def generate_time_series_bakery_data():
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(730)]
    
    data = pd.DataFrame({'datum': dates})
    data['dag_van_de_week'] = data['datum'].dt.weekday
    data['maand'] = data['datum'].dt.month
    data['is_weekend'] = data['dag_van_de_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Simuleer verkoop: weekend is drukker, zomer is rustiger
    base_sales = 100
    weekend_bonus = data['is_weekend'] * 50
    season_effect = np.sin(data['maand'] * (2 * np.pi / 12)) * 20
    noise = np.random.normal(0, 10, len(data))
    
    data['verkoop_target'] = base_sales + weekend_bonus + season_effect + noise
    return data

df = generate_time_series_bakery_data()
# Features voor het model (datum zelf kan niet in het model, wel de afgeleiden)
X = df[['dag_van_de_week', 'maand', 'is_weekend']]
y = df['verkoop_target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Training (We gebruiken XGBoost als kampioen)
model = XGBRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 3. PROTOTYPE FUNCTIE: Van Datum naar Advies
def geef_bakkerij_advies(test_datum_str):
    try:
        test_datum = datetime.strptime(test_datum_str, '%Y-%m-%d')
        
        # Features extraheren uit de gekozen datum
        input_data = pd.DataFrame({
            'dag_van_de_week': [test_datum.weekday()],
            'maand': [test_datum.month],
            'is_weekend': [1 if test_datum.weekday() >= 5 else 0]
        })
        
        # Voorspelling
        voorspelde_verkoop = model.predict(input_data)[0]
        
        # Personeelsadvies (Heuristiek: 1 medewerker per 45 broden)
        benodigd_personeel = int(np.ceil(voorspelde_verkoop / 45))
        
        print(f"\n--- RAPPORT VOOR {test_datum_str} ---")
        print(f"Verwachte drukte: {'Hoog' if voorspelde_verkoop > 120 else 'Normaal'}")
        print(f"Voorspelde verkoop: {voorspelde_verkoop:.0f} eenheden")
        print(f"Geadviseerd personeel: {benodigd_personeel} medewerkers")
        print("-" * 30)
        
    except ValueError:
        print("Gebruik formaat YYYY-MM-DD")

# 4. UITVOERING
print("POC Model getraind op tijdreeksdata.")
# Simuleer een interactie
geef_bakkerij_advies('2026-05-23') # Een zaterdag in de toekomst
geef_bakkerij_advies('2026-05-26') # Een dinsdag in de toekomst