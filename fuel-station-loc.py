import numpy as np
import pandas as pd
import requests

# AN Bing Maps API key
bing_maps_api_key = "AkG58SoxLujRYGeH30ePGMT_9gz3aQ6UCiGBn3EeZ5BO6d-2OYlos9L1yH0gbopC"

# Find longitude and latitude using country code and postal code using Bing Maps API
def find_location(row):
    base_url = "http://dev.virtualearth.net/REST/v1/Locations"

    params = {
        "key": bing_maps_api_key,
        "countryRegion": row["Country code"],
        "postalCode": row["Postal Code"],
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()

        if data["resourceSets"][0]["estimatedTotal"] > 0:
            location = data["resourceSets"][0]["resources"][0]["point"]["coordinates"]
            longitude, latitude = location

            # Return latitude and longitude
            return pd.Series({"Latitude": latitude, "Longitude": longitude})
        else:
            return pd.Series({"Latitude": None, "Longitude": None})
    else:
        return pd.Series({"Latitude": None, "Longitude": None})

if __name__ == "__main__":
    excel_file_path = "Travel Matrix.xlsx" # Enter filepath here
    df = pd.read_excel(excel_file_path,sheet_name="Sheet2 (2)",usecols="A:G",nrows=12) # Missing: build logic for last row with non empty postal code
    # Apply the function
    df[["Latitude", "Longitude"]] = df.apply(find_location, axis=1)
    print(df)
