import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import holidays
import os

# Machine Learning Modellen
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- CONFIGURATIE & UI STYLE ---
st.set_page_config(page_title="Bakkerij AI Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] { color: #31333F !important; font-weight: bold; }
    [data-testid="stMetricValue"] { color: #1f77b4 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA INLADEN & MULTI-MODEL TRAINING ---
@st.cache_data
def load_and_evaluate():
    csv_path = './ml_research/data/data.csv'
    if not os.path.exists(csv_path):
        st.error(f"Bestand niet gevonden op {csv_path}. Run eerst je generate_csv.py script!")
        return None, None, None

    df = pd.read_csv(csv_path)
    df['datum'] = pd.to_datetime(df['datum'])
    
    # Feature selectie (moet matchen met generate_csv.py)
    features = ['dag_van_de_week', 'maand', 'is_weekend', 'is_feestdag']
    X = df[features]
    y = df['verkoop_target']
    
    # Split data voor evaluatie (laatste 20% als testset)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    models = {
        "Baseline (Linear)": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    metrics_results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Bereken metrics voor het onderzoeksgedeelte
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        bias = np.mean(preds - y_test)
        
        metrics_results.append({
            "Model": name, 
            "MAE": round(mae, 2), 
            "RMSE": round(rmse, 2), 
            "BIAS": round(bias, 2)
        })
        trained_models[name] = model

    return trained_models, pd.DataFrame(metrics_results), df

# Initialiseer data en modellen
models_dict, metrics_df, historical_df = load_and_evaluate()

# Stop de uitvoering als de data niet gevonden is
if models_dict is None:
    st.stop()

# --- 2. TABS STRUCTUUR ---
tab1, tab2 = st.tabs(["📊 Operationele Planning", "🧪 Model Performance"])

# Belgische feestdagen generator
be_holidays = holidays.BE(years=[2024, 2025, 2026])

# --- TAB 1: PLANNING (GEBRUIKERSINTERFACE) ---
with tab1:
    st.title("🍞 Dagelijkse Planning voor de Bakker")
    
    col_side, col_main = st.columns([1, 3])
    
    with col_side:
        st.subheader("⚙️ Instellingen")
        gekozen_datum = st.date_input("Selecteer datum", datetime(2026, 5, 23))
        productiviteit = st.slider("Productiviteit (units per bakker)", 30, 60, 45)
        
        # Kies welk model de voorspelling doet
        gekozen_model_name = st.selectbox(
            "Selecteer ML-Model", 
            list(models_dict.keys()), 
            index=2 # Standaard XGBoost
        )

    # Voorspelling Logica voor geselecteerde dag
    is_f = 1 if gekozen_datum in be_holidays else 0
    input_data = pd.DataFrame({
        'dag_van_de_week': [gekozen_datum.weekday()],
        'maand': [gekozen_datum.month],
        'is_weekend': [1 if gekozen_datum.weekday() >= 5 else 0],
        'is_feestdag': [is_f]
    })
    
    selected_model = models_dict[gekozen_model_name]
    voorspelling = selected_model.predict(input_data)[0]
    
    # Bereken personeel (0 als de bakker gesloten is/voorspelling nihil is)
    personeel = int(np.ceil(voorspelling / productiviteit)) if voorspelling > 10 else 0

    with col_main:
        # KPI kaarten
        c1, c2, c3 = st.columns(3)
        c1.metric("Voorspelde Verkoop", f"{voorspelling:.0f} units")
        c2.metric("Benodigd Personeel", f"{personeel} medewerkers")
        
        # Status bepaling
        if voorspelling < 10:
            dag_status = "🔴 Gesloten (Maandag)"
        elif is_f:
            dag_status = "🎉 Feestdag (Extra Drukte)"
        else:
            dag_status = "🍞 Normale Werkdag"
        c3.metric("Status van de Dag", dag_status)

        st.divider()
        
        # --- GRAFIEK: WEEK TREND ---
        st.subheader(f"Trendoverzicht rond {gekozen_datum.strftime('%d %B')}")
        
        week_dates = [gekozen_datum + timedelta(days=i) for i in range(-3, 4)]
        week_df = pd.DataFrame({
            'datum': week_dates,
            'dag_van_de_week': [d.weekday() for d in week_dates],
            'maand': [d.month for d in week_dates],
            'is_weekend': [1 if d.weekday() >= 5 else 0 for d in week_dates],
            'is_feestdag': [1 if d in be_holidays else 0 for d in week_dates]
        })
        
        # FIX: Forceer datetime type voor de .dt accessor
        week_df['datum'] = pd.to_datetime(week_df['datum'])
        
        # Voorspel voor de hele week
        week_df['Voorspelling'] = selected_model.predict(week_df[['dag_van_de_week', 'maand', 'is_weekend', 'is_feestdag']])
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.set_style("whitegrid")
        
        # Teken lijn (gebruik .dt voor nette labels)
        x_labels = week_df['datum'].dt.strftime('%a %d')
        sns.lineplot(x=x_labels, y='Voorspelling', data=week_df, marker='o', linewidth=2, ax=ax)
        
        # Verticale lijn voor de geselecteerde dag
        ax.axvline(x=gekozen_datum.strftime('%a %d'), color='red', linestyle='--', alpha=0.6, label='Selectie')
        
        ax.set_ylim(0, 250)
        plt.ylabel("Verkochte eenheden")
        st.pyplot(fig)

# --- TAB 2: METRICS (ONDERZOEKSINTERFACE) ---
with tab2:
    st.title("🧪 Model Validatie & Vergelijking")
    st.markdown("""
        In dit tabblad worden de prestaties van verschillende Machine Learning algoritmen vergeleken. 
        Dit vormt de kwantitatieve basis voor de modelselectie in de Bachelorproef.
    """)

    # Tabel met alle metrics
    st.subheader("Vergelijkende Tabel")
    st.dataframe(metrics_df.set_index('Model'), use_container_width=True)

    # Visualisatie van de foutmarge (MAE)
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.subheader("Mean Absolute Error (MAE)")
        fig_mae, ax_mae = plt.subplots()
        sns.barplot(x='Model', y='MAE', data=metrics_df, palette='magma', ax=ax_mae)
        ax_mae.set_ylabel("Gemiddelde afwijking (units)")
        st.pyplot(fig_mae)
        st.caption("Hoe lager de MAE, hoe nauwkeuriger het model gemiddeld voorspelt.")

    with col_plot2:
        st.subheader("Model BIAS")
        fig_bias, ax_bias = plt.subplots()
        sns.barplot(x='Model', y='BIAS', data=metrics_df, palette='coolwarm', ax=ax_bias)
        ax_bias.axhline(0, color='black', linewidth=0.8)
        ax_bias.set_ylabel("Systematische afwijking")
        st.pyplot(fig_bias)
        st.caption("Positieve BIAS = overschatting, Negatieve BIAS = onderschatting.")

    with st.expander("🎓 Interpretatie voor Bachelorproef"):
        st.write("""
            *   **Baseline (Linear):** Dient als nulmeting. Als complexere modellen niet beter presteren, geniet dit model de voorkeur door zijn eenvoud.
            *   **Random Forest:** Een ensemble model dat goed omgaat met niet-lineaire verbanden.
            *   **XGBoost:** Vaak het best presterende model voor gestructureerde data (tabellen), maar gevoeliger voor hyperparameters.
        """)