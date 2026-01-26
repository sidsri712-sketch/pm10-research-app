import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tempfile
import os

# Page Settings
st.set_page_config(page_title="PM10 Spatial ML Tool", layout="wide")

# --- APP HEADER ---
st.title("🔬 PM10 Spatial Analysis & Machine Learning Tool")
st.markdown("""
This software integrates **Environmental Sensor Data** with **GIS Shapefiles** to predict PM10 concentrations using a Random Forest Machine Learning model.
""")

# --- IN-APP DOCUMENTATION ---
with st.expander("📖 READ THIS FIRST: How to Prepare Your Files", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 1. CSV Data Requirements")
        st.write("Your Excel/CSV must have these **exact** column headers:")
        st.code("location_id, traffic, hour_type, pm10")
        st.caption("Note: 'hour_type' must contain only 'peak' or 'offpeak'.")
    
    with col2:
        st.markdown("### 2. GIS Shapefile Requirements")
        st.write("You must upload **three specific files** together:")
        st.write("• `.shp` (Geometry)")
        st.write("• `.shx` (Index)")
        st.write("• `.dbf` (Attributes/ID)")

st.divider()

# --- SIDEBAR CONTROL ---
st.sidebar.header("Step 1: Data Input")

# File Uploaders in Sidebar for a cleaner look
csv_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
shp_files = st.sidebar.file_uploader("Upload Shapefile Set (.shp, .shx, .dbf)", 
                                     type=["shp", "shx", "dbf"], 
                                     accept_multiple_files=True)

# --- MAIN ANALYSIS LOGIC ---
if csv_file and shp_files:
    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        
        # Validation
        required_cols = ["location_id", "traffic", "hour_type", "pm10"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Column mismatch! Ensure your CSV has: {required_cols}")
        else:
            # Temporary storage for Shapefile components
            with tempfile.TemporaryDirectory() as tmpdir:
                for uploaded_file in shp_files:
                    with open(os.path.join(tmpdir, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Load GIS data
                shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
                gdf = gpd.read_file(shp_path)

            # 1. Merge & Encode
            data = gdf.merge(df, on="location_id")
            data["hour_type_num"] = data["hour_type"].map({"offpeak": 0, "peak": 1})

            # 2. Machine Learning Pipeline
            X = data[["traffic", "hour_type_num"]]
            y = data["pm10"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            data["Predicted_PM10"] = model.predict(X)
            y_pred = model.predict(X_test)

            # --- DISPLAY RESULTS ---
            st.success("✅ Model Training & Spatial Mapping Successful!")
            
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Model R² Score", f"{r2_score(y_test, y_pred):.3f}")
            m2.metric("RMSE (Error)", f"{mean_squared_error(y_test, y_pred, squared=False):.2f} µg/m³")
            m3.metric("Samples Processed", len(data))

            st.divider()

            # Map and Analysis Row
            left_col, right_col = st.columns([2, 1])

            with left_col:
                st.subheader("🗺 Spatial Concentration Map")
                fig, ax = plt.subplots(figsize=(10, 8))
                data.plot(column="Predicted_PM10", cmap="YlOrRd", legend=True, 
                          ax=ax, markersize=120, edgecolor="black", alpha=0.8)
                ax.set_axis_off()
                st.pyplot(fig)
                st.caption("Figure 1: Predicted PM10 levels across the study area based on the Random Forest model.")

            with right_col:
                st.subheader("📊 Key Driver Analysis")
                # Feature Importance
                importances = model.feature_importances_
                imp_df = pd.DataFrame({"Factor": ["Traffic", "Time of Day"], "Importance": importances})
                st.bar_chart(imp_df.set_index("Factor"))
                st.write("This chart indicates which variable had the most significant impact on the PM10 predictions.")

            st.divider()
            
            # Data Export for Paper
            st.subheader("📥 Export Results for Manuscript")
            csv_export = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediction Data (CSV)",
                data=csv_export,
                file_name="pm10_model_results.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"🚨 Error: {e}")
        st.info("Tip: Ensure 'location_id' in your CSV matches the ID field in your Shapefile.")

else:
    # Instructions shown when no files are uploaded
    st.info("👋 Welcome! Please upload your data files in the sidebar to generate the spatial analysis.")
    
    # Show an example of what the data should look like
    st.subheader("Example CSV Format")
    example_df = pd.DataFrame({
