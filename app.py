import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import geopandas as gpd

st.set_page_config(page_title="Amity Lucknow PM10 Analysis", layout="wide")

st.title("📍 Amity Lucknow Campus: PM10 Spatial Mapper")

# 1. File Upload
uploaded_file = st.sidebar.file_uploader("Upload your Research CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Cleaning column names to avoid "Space" errors
    df.columns = [c.strip() for c in df.columns]
    
    st.write("### Data Preview (First 5 Rows)")
    st.dataframe(df.head())

    # 2. Identify Columns
    # We use your exact headers from the Amity file
    target = 'PM10'
    feature = 'traffic volume in peak hours'
    
    # 3. Handle Missing Data (The Fix)
    # Split data: 'train' has PM10 values, 'predict' is the empty grid
    train_df = df.dropna(subset=[target, feature])
    predict_df = df[df[target].isna()].copy()

    if len(train_df) > 0:
        st.success(f"Model Training: Found {len(train_df)} sensor readings.")
        
        # 4. Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train_df[[feature]], train_df[target])
        
        # 5. Predict for the empty rows
        if len(predict_df) > 0:
            predictions = model.predict(predict_df[[feature]])
            df.loc[df[target].isna(), target] = predictions
            st.info(f"Predictions complete for {len(predict_df)} grid points.")

        # 6. Visualization
        st.subheader("📊 Spatial PM10 Heatmap")
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Create a scatter map using Lat/Lon
        scatter = ax.scatter(df['longitude'], df['latitude'], 
                             c=df[target], cmap='YlOrRd', s=15, alpha=0.8)
        
        plt.colorbar(scatter, label='PM10 Concentration (µg/m³)')
        ax.set_title("Amity University Lucknow: Predicted PM10 Distribution")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        st.pyplot(fig)
        
        # 7. Download Final Data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Final Research CSV", csv, "Amity_PM10_Final.csv", "text/csv")
        
    else:
        st.error("Error: No PM10 values found in the CSV. The model needs at least some data to learn from!")

else:
    st.warning("Please upload the 'amity_campus_research_data.csv' to see the map.")
