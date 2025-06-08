import geopandas as gpd
import folium
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Function to Select Shapefile
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide root window
    file_path = filedialog.askopenfilename(title="Select a GIS Data File", filetypes=[("Shapefiles", "*.shp"), ("CSV Files", "*.csv")])
    return file_path

# Load Geospatial Data
def load_data(file_path):
    try:
        if file_path.endswith(".shp"):
            data = gpd.read_file(file_path)
        elif file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please select a .shp or .csv file.")
        print(f"Loaded Data:\n{data.head()}")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Train Predictive Model for Mining Expansion Risks
def train_model(data):
    if "Latitude" in data.columns and "Longitude" in data.columns:
        # Creating dummy risk factor based on location
        data["RiskFactor"] = np.random.randint(0, 2, size=len(data))

        # Selecting Features and Labels
        X = data[["Latitude", "Longitude"]]
        y = data["RiskFactor"]

        # Splitting Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling Features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Model
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        # Predict Expansion Risks
        predictions = model.predict(X_test_scaled)
        print("Predicted Expansion Risks:\n", predictions)
        return model
    else:
        print("Missing geospatial coordinates in data.")
        return None

# Visualize Mining Sites Using Folium
def plot_map(data):
    if "Latitude" in data.columns and "Longitude" in data.columns:
        # Define Map Center
        map_center = [data["Latitude"].mean(), data["Longitude"].mean()]
        mining_map = folium.Map(location=map_center, zoom_start=6)

        # Add Mining Locations
        for _, row in data.iterrows():
            folium.Marker([row["Latitude"], row["Longitude"]], popup=f"Site: {row.get('SiteName', 'Unknown')}").add_to(mining_map)

        # Show Map
        mining_map.save("illegal_mining_map.html")
        print("Map saved as 'illegal_mining_map.html'. Open it in your browser to view.")
    else:
        print("Missing geospatial coordinates in data.")

# Main Execution
if __name__ == "__main__":
    file_path = select_file()
    
    if file_path:
        data = load_data(file_path)
        
        if data is not None:
            train_model(data)  # Train prediction model
            plot_map(data)  # Generate interactive map
        else:
            print("Failed to process data file.")
    else:
        print("No file selected.")
