#Only me, datascientist and ml enginner can understand this code
#This is not beginner friendly!!!!!!!!!!!!

import pandas as pd
from lightgbm import LGBMRegressor
import plotly.express as px

x_data = pd.read_csv("combine.csv")


if "country" in x_data.columns:
    x_data = x_data[~x_data["country"].str.lower().str.contains("western sahara", na=False)]


train_features = [f"BCA.ind.{year}" for year in range(1995, 2015)] 
X_train = x_data[train_features]
y_train = x_data["BCA.ind.2022"]  


future_features = [f"BCA.ind.{year}" for year in range(2003, 2023)] 
X_future = x_data[future_features]
X_future.columns = X_train.columns


model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
model.fit(X_train, y_train)

predictions_2030 = model.predict(X_future)
x_data["BCA.ind.2030_predicted"] = [min(100, max(0, val)) for val in predictions_2030]

# 7. DEDUPLICATION: Group by country/iso code to flatten duplicates (like France)
df_grouped = x_data.groupby(["country", "iso"], as_index=False)["BCA.ind.2030_predicted"].mean()


def assign_tier(val):
    if 80 <= val <= 100: return "80-100: Low Risk (Yellow)"
    elif 50 <= val < 80: return "50-80: Moderate Risk (Orange)"
    elif 30 <= val < 50: return "30-50: Elevated Risk (Red)"
    else: return "0-30: Critical Risk (Deep Slate)"

df_grouped["Risk_Tier"] = df_grouped["BCA.ind.2030_predicted"].apply(assign_tier)


color_map = {
    "80-100: Low Risk (Yellow)": "#f1c40f",
    "50-80: Moderate Risk (Orange)": "#e67e22",
    "30-50: Elevated Risk (Red)": "#e74c3c",
    "0-30: Critical Risk (Deep Slate)": "#2c3e50"
}


fig = px.choropleth(
    df_grouped,
    locations="iso",                     
    color="Risk_Tier",                   
    color_discrete_map=color_map,        
    hover_name="country",                
    hover_data={"BCA.ind.2030_predicted": ":.2f"}, 
    title="<b>Figure 4.2: Global Risk Projections Map (2030)</b><br><sup>LightGBM Predicted Black Carbon Performance Categories</sup>",
    projection="natural earth",         
    category_orders={"Risk_Tier": ["80-100: Low Risk (Yellow)", "50-80: Moderate Risk (Orange)", "30-50: Elevated Risk (Red)", "0-30: Critical Risk (Deep Slate)"]}
)


fig.update_geos(
    showcountries=True, 
    countrycolor="#ffffff",    
    showframe=False, 
    showcoastlines=True, 
    coastlinecolor="#bdc3c7"
)

fig.update_layout(
    margin={"r":10, "t":80, "l":10, "b":10},
    legend=dict(title="<b>EPI Risk Tiers</b>", orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
)


fig.show()
