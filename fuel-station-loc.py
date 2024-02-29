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

def new_node(ro,tm,stn): #ro is the truck info, tm is the travel matrix, stn is the remaining fuel in stations
    dest_node = np.array([tm.iloc[12:,0]]).flatten().tolist()
    probabilities = np.array([0.1,0.1,0.07,0.05,0.05,0.13,0.3,0.2])
    if ro['Intermittent Stop'] != 0: #else take the intermitten stop as the starting stop
        ''' #refuels the truck and subtracts fuel from station
        fuel_left = ro["Remaining Fuel"] - distance_between_nodes(start_dest[i-1,0],end_dest[i-1,0],main_excel_df)
        ro["Travel Time"] += ro["Remaining Time"]
        fuel_needed = 60 - ro["Remaining Fuel"]
        station = ro["Intermittent Stop"] - 1
        stn[station] -= fuel_needed
        ro["Remaining Fuel"] = 60
        ro["Remaining Range"] = 700
        '''
        ro['Starting Node'] = ro['Intermittent Stop']
        ro['Intermittent Stop'] = 0
        #remaining travel time function here
        
    else: #if stop == 0 -> let ending node be the starting node and find a new ending node
        ro['Starting Node'] = ro['Ending Node']
        ro["Travel Time"] += ro["Remaining Time"]
        starts = np.ones(len(dest_node))
        starts *= ro["Starting Node"]
        dist_check = distance_between_nodes(starts,dest_node,main_excel_df)
        dist_check = dist_check[dist_check < 13-ro["Travel Time"]] #will be in time units
        if len(dist_check)>0:
            #complete -> set time to be 1000000, will not be called again
            ro["Remaining Time"] = 1000000
            return ro,stn
        
        end_node = np.random.choice(dest_node, p = probabilities)
        while end_node == ro["Starting Node"]:
            end_node = np.random.choice(dest_node, p = probabilities)
        ro["Ending Node"] = end_node
        '''
        if ro["Remaining Range"] - distance_between_nodes(ro["Starting Node"],ro["Ending Node"],main_excel_df) <= 700*0.25:
            inter_stat = tm.iloc[ro["Starting Node"]-1,ro["Starting Node"]] == tm.iloc[ro["Starting Node"],ro["Ending Node"]] #returns a row or true and false to see which stations are compatible
            col_ind = np.where(inter_stat)[1]
            col_ind = col_ind[con_ind < 12] #filters for the compatable refuelling stations
            for j in col_ind:
                rem_rnge = ro["Remaining Range"] - distance_between_nodes(ro["Starting Node"],j,main_excel_df)
                interp_rnge = np.append(interp_rnge, rem_rnge)
            interp_stn = col_ind[np.abs(interp_rnge - (700*25)).argmin()] #obtains the intermittent station number
            ro["Intermittent Stop"] = interp_stn
            #remaining travel time function here
        '''
    #calcuate distance and time required    
    return ro,stn

def system_wide_sim(tm):
    title = np.array(["Truck #","Starting Node","Ending Node","Intermittent Stop","Remaining Range","Remaining Fuel","Remaining Time", "Travel Time"])
    truck_num = np.arange(1,151)
    truck_num = np.array(truck_num).reshape(150,1)
    tube_num = np.array([401,402,403,404,405,406])
    tube_num = np.array(tube_num).reshape(6,1)
    trav_time = np.zeros((156,1))
    rem_time = np.ones((156,1))
    start_dest = np.ones((156,1))
    end_dest = np.ones((156,1))
    fuel = np.ones((156,1))
    rnge = np.ones((156,1))
    intermitten = np.zeros((156,1))
    dest_node = np.array([tm.iloc[12:,0]]).flatten().tolist()
    probabilities = np.array([0.1,0.1,0.07,0.05,0.05,0.13,0.3,0.2])
    
    for i in truck_num:
        interp_rnge = np.array([])
        interp_stn = 0
        start_dest[i-1,0] *= random.randint(13,tm.iloc[-1,0])
        end_node = np.random.choice(dest_node, p = probabilities)
        while end_node == start_dest[i-1,0]:
            end_node = np.random.choice(dest_node, p = probabilities)
        end_dest[i-1,0] *= end_node
        fuel[i-1,0] = random.uniform(30,60)
        rnge[i-1,0] = 700 * fuel[i-1,0]/60
        ''' -> awaiting function for distance_between_nodes to work
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
        interp_stn = 0
        
    start_dest[150,0] = start_dest[151,0] = start_dest[152,0] = 3 #this could lead to trouble later
    start_dest[153,0] = start_dest[154,0] = start_dest[155,0] = 8
    
    #remaining time of travel function here
    
    combined_matrix = np.vstack((truck_num,tube_num))
    combined_matrix = np.hstack((combined_matrix,start_dest,end_dest,intermitten,rnge,fuel,rem_time,trav_time))
    combined_matrix = np.vstack((title,combined_matrix))
    combined_matrix = pd.DataFrame(combined_matrix)
    combined_matrix.columns = combined_matrix.iloc[0]
    combined_matrix = combined_matrix.drop(0)
    
    return combined_matrix

def sim_by_time(sys_all,tm): #takes the entire starting matrix to start simulation
    stat_mtrx = np.arange(1,157)
    plc_hldr = np.array([])
    for i in range (1,13):
        plc_hldr = np.append(plc_hldr, "Station " + i)
    stat_mtrx = np.hstack((stat_mtrx,plc_hldr)) #establishes title row of the simulation
    stat_mtrx = np.insert(stat_mtrx, 0, "Time")
    stat_mtrx = pd.DataFrame(stat_mtrx)
    stat_mtrx.columns = stat_mtrx.iloc[0]
    stat_mtrx = stat_mtrx.drop(0)
    init_stn = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
    
    times_range = pd.date_range(start="7:00", end="21:00", freq='10T').T
    formatted_times = times_range.strftime('%H:%M')
    travel_time = np.array([sys_all["Remaining Time"]])
    old_time = travel_time
    for i in formatted_times:
        if i == "7:00":
            ro = np.hstack((i,travel_time,init_stn))
            stat_mtrx.loc[len(stat_mtrx)] = ro
            continue
        new_time = old_time - pd.Timedelta(minutes=10)
        time_check = new_time[new_time <= 0]
        if len(time_check) > 0:
            new_loc = np.where(new_time < 0)[0] #returns a row of array indexes that arrived
            for j in new_loc:
                row_data = sys_all.iloc[j] #go to the jth row of sys_all -> truck info
                row_data,init_stn = new_node(row_data,tm,init_stn)#put into new node function
                sys_all.iloc[j] = row_data
                #take output and take the new remaining time into the new_time slot
        
    return None
    
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
    sim = sim_by_time(sys,travel_matrix)