import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print("Leyendo Excel...")
df = pd.read_excel(
    "Leads Mexico Enero 2022 - Enero 13 2026 excel.xlsx",
    usecols=[1, 11, 14, 15],  # Fecha, Plataforma, Cualificacion, Sub-Cualificacion
    header=0
)
df.columns = ["fecha", "plataforma", "cual", "subcual"]

df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
df["anio"]  = df["fecha"].dt.year
df["mes"]   = df["fecha"].dt.month

df25 = df[df["anio"] == 2025].copy()
print(f"2025 total: {len(df25):,}\n")

# Analisis Plataforma Chatbot
bot_platform = "MX_LEAD_CHATBOT_QUALIF"
is_bot = df25["plataforma"].astype(str).str.contains("CHATBOT", case=False, na=False)
print(f"Leads de chatbot en 2025: {is_bot.sum():,}")

# Analisis de nulos en cualificacion
sin_cual = df25["cual"].isna()
print(f"Leads sin cualificacion (NaN) en 2025: {sin_cual.sum():,}")

# Cruzar bot vs sin cualificacion
bot_y_sincual = (is_bot & sin_cual).sum()
nobot_y_sincual = (~is_bot & sin_cual).sum()
print(f"  - De los NaN, cuantos son chatbot? {bot_y_sincual:,}")
print(f"  - De los NaN, cuantos NO son chatbot? {nobot_y_sincual:,}\n")

# Analisis de Sub-Cualificacion para "Hot Premium"
print("=== Valores en Sub-Cualificacion que contienen 'Hot' o 'Premium' ===")
df25["subcual"] = df25["subcual"].astype(str).str.strip()
hot_prem_mask = df25["subcual"].str.contains("hot|premium", case=False, na=False)
print(df25[hot_prem_mask]["subcual"].value_counts(dropna=False).to_string())

# Ver si hay leads no-bot sin cualificar en Enero-Febrero-Marzo
print("\n=== Distribucion de Leads NO BOT, SIN CUALIFICAR por mes ===")
df_nobot_sincual = df25[~is_bot & sin_cual]
print(df_nobot_sincual["mes"].value_counts().sort_index().to_string())
