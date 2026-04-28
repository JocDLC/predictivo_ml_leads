import pandas as pd

df = pd.read_csv('data/processed/leads_cleaned.csv')
total_global = df['target'].mean() * 100

# Conversion por vehiculo
veh = df.groupby('vehiculo_interes')['target'].agg(['mean','count','sum']).reset_index()
veh.columns = ['vehiculo', 'tasa_conv', 'total_leads', 'hot_leads']
veh['tasa_conv'] = (veh['tasa_conv'] * 100).round(1)
veh = veh.sort_values('total_leads', ascending=False)

print(f'Tasa global de conversion: {total_global:.1f}%')
print()
print(veh.to_string(index=False))
print()

kwid = veh[veh['vehiculo'] == 'KWID'].iloc[0]
pct_del_total = kwid['total_leads'] / len(df) * 100
print(f"KWID: {int(kwid['total_leads'])} leads ({pct_del_total:.1f}% del total)")
print(f"KWID conversion: {kwid['tasa_conv']}% vs global {total_global:.1f}%")
print(f"KWID hot leads: {int(kwid['hot_leads'])} de {int(kwid['total_leads'])}")

# Por canal (origen) dentro de KWID
print()
print("=== KWID desglosado por canal (origen_creacion) ===")
kwid_df = df[df['vehiculo_interes'] == 'KWID']
canal = kwid_df.groupby('origen_creacion')['target'].agg(['mean','count']).reset_index()
canal.columns = ['canal', 'conversion', 'leads']
canal['conversion'] = (canal['conversion'] * 100).round(1)
canal = canal.sort_values('leads', ascending=False)
print(canal.to_string(index=False))

# Por campana dentro de KWID
print()
print("=== KWID desglosado por campana ===")
camp = kwid_df.groupby('campana')['target'].agg(['mean','count']).reset_index()
camp.columns = ['campana', 'conversion', 'leads']
camp['conversion'] = (camp['conversion'] * 100).round(1)
camp = camp.sort_values('leads', ascending=False).head(10)
print(camp.to_string(index=False))
