import numpy as np
import pandas as pd
import requests
import json
import random

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
            latitude, longitude = location

            # Return latitude and longitude
            return pd.Series({"Latitude": latitude, "Longitude": longitude})
        else:
            return pd.Series({"Latitude": None, "Longitude": None})
    else:
        return pd.Series({"Latitude": None, "Longitude": None})

def find_travel_time_distance(start_lat,start_long,end_lat,end_long):
	start_loc = str(start_lat) + "," + str(start_long)
	end_loc = str(end_lat) + "," + str(end_long)
	print(start_loc)
	newurl = f'http://dev.virtualearth.net/REST/v1/Routes/DistanceMatrix?origins={start_loc}&destinations={end_loc}&travelMode=driving&key={bing_maps_api_key}'
	response = requests.get(newurl)
	data = response.json()
	travel_distance = data['resourceSets'][0]['resources'][0]['results'][0]['travelDistance']
	travel_time = data['resourceSets'][0]['resources'][0]['results'][0]['travelDuration']
	return [travel_distance,travel_time]

def system_wide_sim(tm):
    title = np.array(["Truck #","Starting Node","Ending Node","Intermittent Stop","Remaining Range","Remaining Fuel"])
    truck_num = np.arange(1,151)
    truck_num = np.array(truck_num).reshape(150,1)
    tube_num = np.array([401,402,403,404,405,406])
    tube_num = np.array(tube_num).reshape(6,1)
    start_dest = np.ones((156,1))
    end_dest = np.ones((156,1))
    fuel = np.ones((156,1))
    rnge = np.ones((156,1))
    intermitten = np.zeros((156,1))
    dest_node = np.array([tm.iloc[12:,0]]).flatten().tolist()
    probabilities = np.array([0.1,0.1,0.07,0.05,0.05,0.13,0.3,0.2])
    
    for i in truck_num:
        interp_rnge = np.array([])
        start_dest[i-1,0] *= random.randint(13,tm.iloc[-1,0])
        end_node = np.random.choice(dest_node, p = probabilities)
        while end_node == start_dest[i-1,0]:
            end_node = np.random.choice(dest_node, p = probabilities)
        end_dest[i-1,0] *= end_node
        fuel[i-1,0] = random.uniform(30,60)
        rnge[i-1,0] = 700 * fuel[i-1,0]/60
        '''
        if rnge[i-1,0] - distance_between_nodes(start_dest[i-1,0],end_dest[i-1,0],main_excel_df) <= 700*0.25:
            inter_stat = tm.iloc[start_dest[i-1,0]-1,start_dest[i-1,0]] == tm.iloc[start_dest[i-1,0],end_dest[i-1,0]] #returns a row or true and false to see which stations are compatible
            col_ind = np.where(inter_stat)[1]
            col_ind = col_ind[con_ind < 12] #filters for the compatable refuelling stations
            for j in col_ind:
                rem_rnge = rnge[i-1,0] - distance_between_nodes(start_dest[i-1,0],j,main_excel_df)
                interp_rnge = np.append(interp_rnge, rem_rnge)
            interp_stn = col_ind[np.abs(interp_rnge - (700*25)).argmin()] #obtains the intermittent station number
        intermitten[i-1] = interp_stn
        '''
        
    start_dest[150,0] = start_dest[151,0] = start_dest[152,0] = 3
    start_dest[153,0] = start_dest[154,0] = start_dest[155,0] = 8
      
    combined_matrix = np.vstack((truck_num,tube_num))
    combined_matrix = np.hstack((combined_matrix,start_dest,end_dest,intermitten,rnge,fuel))
    combined_matrix = np.vstack((title,combined_matrix))
    combined_matrix = pd.DataFrame(combined_matrix)
    
    return combined_matrix

def sim_by_time(sys_all):

if __name__ == "__main__":
    excel_file_path = "Travel Matrix.xlsx" # Enter filepath here
    df = pd.read_excel(excel_file_path,sheet_name="Addresses",usecols="A:G",nrows=22) # Missing: build logic for last row with non empty postal code
    # Apply the function
    df[["Latitude", "Longitude"]] = df.apply(find_location, axis=1)
    #print(df)
    tester = find_travel_time_distance(df.iloc[5,7],df.iloc[5,8],df.iloc[18,7],df.iloc[18,8]) #this function can be used to extract time and distance -> to work on it more
    travel_matrix = pd.read_excel(excel_file_path,sheet_name="Sheet2",usecols="A:U",nrows=22)
    sys = system_wide_sim(travel_matrix)
    sys.to_excel("tester.xlsx",index=False) #adjust file path here