import numpy as np
import pandas as pd
import random
import os
from datetime import datetime

def new_node(ro,tm,dm,tim,stn,consumption_amt,truck_count,refuelled_truck_count): #ro is the truck info, tm is the travel matrix, stn is the remaining fuel in stations, consumption_amt os fuel consumed at station, truck count is number of trucks visited at that given point in time (non cumulative across times, cumulative for given time), refuelled_truck_count is number of trucks successfully refueled out of the ones that visited at that given point in time

    dest_node = tm.iloc[12:,0].values
    # dest_node = np.array([tm.iloc[12:,0]]).flatten().tolist()
    probabilities = np.array([0.1,0.1,0.07,0.05,0.05,0.13,0.3,0.2])
    interp_rnge = np.array([])
    if int(ro['Intermittent Stop']) != 0: #else take the intermitten stop as the starting stop
        fuel_left = ro["Remaining Fuel"] - dm.iloc[int(ro["Starting Node"])-1,int(ro["Intermittent Stop"])]/760*50 # Fuel remaining after reaching intermittent stop
        ro["Travel Time"] = ro["Travel Time"] + ro["Remaining Time"] # Time driven
        ro["Travel Range"] = ro["Travel Range"] + dm.iloc[int(ro["Starting Node"])-1,int(ro["Intermittent Stop"])] # Distance driven
        fuel_needed = 50 - fuel_left # Refuelling to full tank i.e. 50 kg
        #print(fuel_needed)
        station = int(ro["Intermittent Stop"]) - 1 # station is station node - 1 for indexing element in following array
        consumption_amt[station] += fuel_needed
        truck_count[station] += 1
        if stn[station] > fuel_needed:
            stn[station] -= fuel_needed
            refuelled_truck_count[station] += 1
            ro["Remaining Fuel"] = 50 # Full tank
            ro["Remaining Range"] = 750 # Full range
            ro["Refuelling Stops"] +=1
            ro['Starting Node'] = int(ro['Intermittent Stop'])
        # need to check again for intermittent stop
            if ro["Remaining Range"] - dm.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])] <= 750*0.45 and (int(ro["Starting Node"])*int(ro["Ending Node"])!=304):
                inter_stat = tm.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])] == tm.iloc[int(ro["Starting Node"])-1] #returns a row or true and false to see which stations are compatible
                col_ind = np.where(inter_stat.values.flatten())[0]
                col_ind = col_ind[col_ind < 13] #filters for the compatable refuelling stations
                for j in col_ind:
                    rem_rnge = ro["Remaining Range"] - dm.iloc[int(ro["Starting Node"])-1,j]
                    if rem_rnge > 0 and dm.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])] > dm.iloc[int(ro["Starting Node"])-1,j]:
                        if (int(ro["Ending Node"]) == 16 and j == 8) or (int(ro["Ending Node"]) == 16 and j == 7):
                            interp_rnge = np.append(interp_rnge,9999)
                        else:
                            interp_rnge = np.append(interp_rnge, rem_rnge)
                    else:
                        interp_rnge = np.append(interp_rnge,9999)
                if len(interp_rnge[interp_rnge<9000]) != 0:
                    try:
                        ro["Intermittent Stop"] = col_ind[(interp_rnge - (0.1*750)).argmin()] #obtains the intermittent station number
                        ro["Remaining Time"] = tim.iloc[int(ro["Starting Node"])-1,int(ro["Intermittent Stop"])]
                    except:#seems ottawa to montreal route is having issues
                        #print(ro["Starting Node"],ro["Ending Node"],ro["Remaining Range"])
                        #print("failed to find a suitable station, the range is enough to go from ottawa to montreal and back")
                        ro["Remaining Time"] = tim.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])]
                else:
                    ro["Remaining Time"] = tim.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])]
                    ro["Intermittent Stop"] = 0
            else:
                ro["Remaining Time"] = tim.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])]
                ro["Intermittent Stop"] = 0
        else:
            # ------------------ Dailed to refuel due to unavailability of hydrogen at fuel station ------------
            ro["Remaining Time"] = 1000000
            ro["Ending Node"] = "Finished"
            return ro,stn,consumption_amt,truck_count,refuelled_truck_count
        
    else: #if stop == 0 -> let ending node be the starting node and find a new ending node
        ro["Travel Time"] = ro["Travel Time"] + ro["Remaining Time"]
        ro["Travel Range"] = ro["Travel Range"] + dm.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])]
        ro["Remaining Range"] = ro["Remaining Range"] - dm.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])]
        ro["Remaining Fuel"] = ro["Remaining Fuel"] - dm.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])]/750*50
        ro['Starting Node'] = int(ro['Ending Node'])
        end_node = np.random.choice(dest_node, p = probabilities)
        time_check = 12*60*60 - ro["Travel Time"] - tim.iloc[int(ro["Starting Node"])-1,end_node]
        counter = 0
        while float(end_node) == float(ro["Starting Node"]) and (counter < 10 or time_check < 0):
            end_node = np.random.choice(dest_node, p = probabilities) # can just pop the origin node from
            time_check = 36000 - ro["Travel Time"] - tim.iloc[int(ro["Starting Node"])-1,end_node] #13h max -> 10h right now
            counter += 1
        # ------------------ SIMULATION EXIT CONDITION ------------
        if counter > 9 or time_check < 0: #end the truck sim if more than 10 fails
            ro["Remaining Time"] = 1000000
            ro["Ending Node"] = "Finished"
            return ro,stn,consumption_amt,truck_count,refuelled_truck_count
        
        ro["Ending Node"] = end_node
        
        if ro["Remaining Range"] - dm.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])] <= 750*0.45 and (int(ro["Starting Node"])*int(ro["Ending Node"])!=304):
            inter_stat = tm.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])] == tm.iloc[int(ro["Starting Node"])-1] #returns a row or true and false to see which stations are compatible
            col_ind = np.where(inter_stat.values)[0]
            col_ind = col_ind[col_ind < 13] #filters for the compatable refuelling stations
            for j in col_ind:
                rem_rnge = ro["Remaining Range"] - dm.iloc[int(ro["Starting Node"])-1,j]
                if rem_rnge > 0 and dm.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])] > dm.iloc[int(ro["Starting Node"])-1,j]:
                    if (int(ro["Ending Node"]) == 16 and j == 8) or (int(ro["Ending Node"]) == 16 and j == 7):
                        interp_rnge = np.append(interp_rnge,9999) # Failure
                    else:
                        interp_rnge = np.append(interp_rnge, rem_rnge)
                else:
                    interp_rnge = np.append(interp_rnge,9999) # Failure
            if len(interp_rnge[interp_rnge<9000]) != 0:
                try:
                    ro["Intermittent Stop"] = col_ind[(interp_rnge - (0.1*750)).argmin()] #obtains the intermittent station number
                    ro["Remaining Time"] = tim.iloc[int(ro["Starting Node"])-1,int(ro["Intermittent Stop"])]
                except:#seems ottawa to montreal route is having issues
                    #print(ro["Starting Node"],ro["Ending Node"],ro["Remaining Range"])
                    #print("failed to find a suitable station, the range is enough to go from ottawa to montreal and back")
                    ro["Remaining Time"] = tim.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])]
            else:
                ro["Remaining Time"] = tim.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])]
                ro["Intermittent Stop"] = 0      
        else:
            ro["Remaining Time"] = tim.iloc[int(ro["Starting Node"])-1,int(ro["Ending Node"])]  
    return ro,stn,consumption_amt,truck_count,refuelled_truck_count

def system_wide_sim(tm,dm,tim):
    title = np.array(["Truck #","Starting Node","Ending Node","Intermittent Stop","Remaining Range","Remaining Fuel","Remaining Time","Travel Time","Travel Range","Refuelling Stops"])
    # Rows are number of vehicles = 150 + 6 tube trailers
    # Variables below are columns =  9 columns
    truck_num = np.arange(1,151,dtype=int).reshape(150,1)
    tube_num = np.array([401,402,403,404,405,406]).reshape(6,1)
    trav_time = np.zeros((156,1))
    rem_time = np.ones((156,1))
    start_dest = np.ones((156,1),dtype=int)
    end_dest = np.ones((156,1),dtype=int)
    fuel = np.ones((156,1))
    rnge = np.ones((156,1))
    trnge = np.zeros((156,1))
    intermitten = np.zeros((156,1),dtype=int)
    intermittent_stop_count = np.zeros((156,1),dtype=int)
    dest_node = tm.iloc[12:,0].values
    probabilities = np.array([0.1,0.1,0.07,0.05,0.05,0.13,0.3,0.2]) # Based on traffic data
    
    for i in truck_num:
        interp_rnge = np.array([])
        interp_stn = 0
        start_dest[i-1,0] *= random.randint(13,tm.iloc[-1,0]) # Can use index to reassign instead of multipl (not an issue)

        # Can remove this check conditions change to remove start destination from dest_node and probabilities
        # but pop/remove alters array since mutable so not necessary
        end_node = np.random.choice(dest_node, p = probabilities)
        while end_node == start_dest[i-1,0]: # Python compares integer to array and returns array([True])
            end_node = np.random.choice(dest_node, p = probabilities)
        end_dest[i-1,0] *= end_node
        fuel[i-1,0] = random.uniform(30,50) # Range of fuel tank between 30 to 50 kg/H2
        rnge[i-1,0] = 750 * fuel[i-1,0]/50 # Distance based on 750km for full tank 50kg/H2 ( US DOE current estimate: 9.4 mile per kg H2 )

        # Finding intermittent stations if needed
        if float(rnge[i-1,0]) - dm.iloc[int(start_dest[i-1,0]-1),int(end_dest[i-1,0])] <= 750*0.45 and (start_dest[i-1,0]*end_dest[i-1,0] != 304): # 304 is 16 * 19 (node nums)
            # Start looking for fuel station when 45% 
            # does not allow toronto <> barrie routes to find intermitten stns
            inter_stat = tm.iloc[int(start_dest[i-1,0]-1),int(end_dest[i-1,0])] == tm.iloc[int(start_dest[i-1,0]-1)] #returns an array of true and false to see which stations are compatible based on service direction 'E' or 'W'

            col_ind = np.where(inter_stat.values)[0]
            col_ind = col_ind[col_ind < 13] # Only look at fuel stations and not destination nodes 
            for j in col_ind: # iterate through all fuel stations either east or west based on service dir
                rem_rnge = float(rnge[i-1,0]) - dm.iloc[int(start_dest[i-1,0]-1),j] # Remaining range = current range - range after reaching fuel station
                if rem_rnge > 0 and dm.iloc[int(start_dest[i-1,0]-1),int(end_dest[i-1,0])] > dm.iloc[int(start_dest[i-1,0]-1),j]:
                    if (end_dest[i-1,0] == 16 and j == 8) or (end_dest[i-1,0] == 16 and j == 7): # Barrie and port Hope and Barrie and Cambridge
                        interp_rnge = np.append(interp_rnge,9999) # If fuel station is farther from destination relative to origin
                    else:
                        interp_rnge = np.append(interp_rnge, rem_rnge)
                else:
                    interp_rnge = np.append(interp_rnge,9999) # If fuel station is farther from destination relative to origin
            if len(interp_rnge[interp_rnge<9000]) != 0:
                try:
                    interp_stn = col_ind[(interp_rnge - (0.1*750)).argmin()] # Refuel at 10%(75 km) of fuel level without making unnecessary stops # ().argmin() returns indices of of minimum value and col_ind[index] returns station node number. Finds nearest station (argmin based on distance)
                    rem_time[i-1,0] = tim.iloc[int(start_dest[i-1,0]-1),interp_stn] # Time driven from origin to destination
                except: # seems ottawa to montreal route is having issues - unaddressed
                    print(start_dest[i-1,0],end_dest[i-1,0],rnge[i-1,0])
                    # print("failed to find a suitable station, the range is enough to go from ottawa to montreal and back")
                    rem_time[i-1,0] = tim.iloc[int(start_dest[i-1,0]-1),int(end_dest[i-1,0])]
            else:
                rem_time[i-1,0] = tim.iloc[int(start_dest[i-1,0]-1),int(end_dest[i-1,0])]  
        else:
            rem_time[i-1,0] = tim.iloc[int(start_dest[i-1,0]-1),int(end_dest[i-1,0])]
        intermitten[i-1] = interp_stn 
        interp_stn = 0 # Do we need this? Will stay 0 if no intermittent stops are needed so dont need to set as 0?

    # Tube trailer network.....    
    start_dest[150,0] = start_dest[151,0] = start_dest[152,0] = 3
    start_dest[153,0] = start_dest[154,0] = start_dest[155,0] = 8
    rem_time[150:,0] = 99999
    
    # Stack all arrays to make dataframe
    combined_matrix = np.vstack((truck_num,tube_num))
    combined_matrix = np.hstack((combined_matrix,start_dest,end_dest,intermitten,rnge,fuel,rem_time,trav_time,trnge,intermittent_stop_count))
    combined_matrix = pd.DataFrame(combined_matrix,columns=title,index=np.arange(1,len(combined_matrix)+1))
    combined_matrix = combined_matrix.astype({"Truck #":'int32',"Starting Node":'int32',"Ending Node":'int32',"Intermittent Stop":'int32',"Refuelling Stops":'int32'})
    
    return combined_matrix

def sim_by_time(sys_all,tm,dm,tim): #takes the entire starting matrix to start simulation
    stat_mtrx = np.arange(1,157) # 150 Trucks + 6 tube trailers
    plc_hldr = np.array([])
    for i in range (1,13): # Label fuel stations
        plc_hldr = np.append(plc_hldr, "Station " + str(i))
    stat_mtrx = np.hstack((stat_mtrx,plc_hldr)) # establishes title row of the simulation
    stat_mtrx = np.insert(stat_mtrx, 0, "Time") # Add column for Time at beginning of matrix
    stat_mtrx = pd.DataFrame([stat_mtrx])
    stat_mtrx.columns = stat_mtrx.iloc[0]
    stat_mtrx = stat_mtrx.drop(0)

    # Separate sheet for consumption of H2 at each fuel station
    consumption_matrix = stat_mtrx.copy(deep=True)
    consumption_matrix = consumption_matrix.iloc[:, [0] + list(range(-12, 0))]

    # Track number of trucks visited at each time so that if station fails to refuel due to unavailaibility of hydrogen, the tube trailer network can be tweaked

    # Separate sheet for number of vehicles at fuel station at any given time
    station_truck_count_tracker = consumption_matrix.copy(deep=True)
    # Separate sheet for number of vehicles successfully refueled at fuel station at any given time
    successfull_refuel_count_tracker = station_truck_count_tracker.copy(deep=True)
    
    init_stn = np.array([50,1700,0,700,800,2200,600,150,1000,1500,200,60]) # Initial Hydrogen amount (kg) at each fuel station
    
    
    h2_tube_trailer = 4600*2-np.sum(init_stn) # 2 sites, each producing 4600kg H2/ day
    times_range = np.arange(0,50401,600).reshape(-1,1) # Time array for 14 hours - should be 13 hours? # 600 second interval between each time point

    travel_time = np.array([sys_all["Remaining Time"]])
    old_time = np.array(travel_time.astype(float)) # already a float - cmn remove this
    for i in range(len(times_range)):
        truck_count = np.zeros(len(init_stn))
        consumption_amt = truck_count.copy()
        refuelled_truck_count = truck_count.copy()
        if i == 0:
            ro = np.hstack((i,old_time.flatten(),init_stn))
            stat_mtrx.loc[len(stat_mtrx)] = ro

            ro = np.hstack((i,truck_count))
            station_truck_count_tracker.loc[len(station_truck_count_tracker)] = ro
            successfull_refuel_count_tracker.loc[len(successfull_refuel_count_tracker)] = ro
            
            ro = np.hstack((i,consumption_amt))
            consumption_matrix.loc[len(consumption_matrix)] = ro
            continue
        new_time = np.array(old_time - 600)
        time_check = new_time[new_time <= 0] # Check if any vehicle has reached its intermittent stop i.e. fuel station (if needed) or destination
        if len(time_check) > 0: # If any vehicle has reached its intermittent stop i.e. fuel station if needed or destination
            new_loc = np.where(new_time <= 0)[1] #returns a row of array indexes that arrived # Truck number = index + 1
            for j in new_loc: # Loop through each truck
                row_data = sys_all.iloc[j] #go to the jth row of sys_all -> truck info
                row_data,init_stn,consumption_amt,truck_count,refuelled_truck_count = new_node(row_data,tm,dm,tim,init_stn,consumption_amt,truck_count,refuelled_truck_count) #put into new node function
                sys_all.iloc[j] = row_data
                new_time[0][j] = row_data["Remaining Time"]
        # take output and take the new remaining time into the new_time slot
        ro = np.hstack((i*600,new_time.flatten(),init_stn))
        stat_mtrx.loc[len(stat_mtrx)] = ro # Add data to next row

        #------------------------------------------- Fix the third hstack for each----------------------------------------
        ro = np.hstack((i*600,consumption_amt))
        consumption_matrix.loc[len(consumption_matrix)] = ro

        ro = np.hstack((i*600,truck_count))
        station_truck_count_tracker.loc[len(station_truck_count_tracker)] = ro

        ro = np.hstack((i*600,refuelled_truck_count))
        successfull_refuel_count_tracker.loc[len(successfull_refuel_count_tracker)] = ro
        old_time = new_time
    return stat_mtrx,sys_all,consumption_matrix,station_truck_count_tracker,successfull_refuel_count_tracker

def save_sim_files():
    current_dir = os.getcwd()
    path1 = os.path.join(current_dir, "Initial-Conditions")
    path2 = os.path.join(current_dir, "Simulation-Data")
    path3 = os.path.join(current_dir, "Final-Conditions")
    path4 = os.path.join(current_dir, "Consumption-Simulation")
    path5 = os.path.join(current_dir, "Station-Reliability")

    os.makedirs(path1, exist_ok = True)
    os.makedirs(path2, exist_ok = True)
    os.makedirs(path3, exist_ok = True)
    os.makedirs(path4, exist_ok = True)
    os.makedirs(path5, exist_ok = True)
    return (path1,path2,path3,path4,path5)

def summary_report(paths):
    # paths[0] = simulation data - time
    # paths[1] = final conditions for each truck 

    # -------------------------Summary 1 - trucks ----------------------
    # Average distance driven by trucks
    # Max distance driven by trucks
    # Minimum distance driven by trucks
    # Average time driven by trucks
    # Max distance driven by trucks
    # Minimum distance driven by trucks
    # Average number of refuelling stops
    truck_dist = []
    truck_time = []
    truck_count_refueling_stops = []

    for file_num,file in enumerate(os.listdir(paths[1])): # Lists all files in the folder and iterates over all
        if file.endswith('.xlsx'): # Check if file is excel file
            file_path = os.path.join(paths[1], file)
            df = pd.read_excel(file_path)
            truck_time.append(df.loc[:149,"Travel Time"].values)
            truck_dist.append(df.loc[:149,"Travel Range"].values)
            truck_count_refueling_stops.append(df.loc[:151,"Refuelling Stops"].values)
    simulations = file_num+1
    truck_dist = [element for row in truck_dist for element in row] # Flatten
    truck_time = [element for row in truck_time for element in row]
    truck_count_refueling_stops = [element for row in truck_count_refueling_stops for element in row]
    max_dist, min_dist, avg_dist = max(truck_dist), min(truck_dist), sum(truck_dist)/len(truck_dist)
    max_time, min_time, avg_time = max(truck_time)/3600, min(truck_time)/3600, sum(truck_time)/(len(truck_time)*3600)
    avg_stops = sum(truck_count_refueling_stops)/len(truck_count_refueling_stops)
    avg_daily_hyd_consump = (avg_dist*50/750)*150 # 50 kg gives 750km, 150 vehicles

    row_labels = ['Average distance travelled','Max. distance','Min. distance','Average travel time (h)','Max. travel time (h)','Min. travel time (h)','Average number of refuelling stops','Max. number of refuelling stops','Avg. daily hydrogen consumption (150 trucks, kg)','Number of simulations run']
    row_vals = [round(avg_dist), round(max_dist), round(min_dist), round(avg_time, 1), round(max_time, 1), round(min_time, 1), round(avg_stops), round(max(truck_count_refueling_stops)),round(avg_daily_hyd_consump,1),simulations]
    pd.DataFrame({'Parameter': row_labels, 'Value': row_vals}).to_excel('Truck-Travel-Summary.xlsx',index=False)
    
    # Sheet 2 - fuel station consumption
    # Rows = fuel stations
    # Total
    # Cols = average consumption(last row avg across all files), max consumption(across all files), min consumption(across all files)
    return


if __name__ == "__main__":
    excel_file_path = "Travel Matrix.xlsx" # Enter filepath here
    travel_matrix = pd.read_excel(excel_file_path,sheet_name="Service-Directions",usecols="A:U",nrows=22)
    distance_matrix = pd.read_excel(excel_file_path,sheet_name="Distance-Matrix",usecols="A:U",nrows=22)
    time_matrix = pd.read_excel(excel_file_path,sheet_name="Driving-Time-Matrix",usecols="A:U",nrows=22)
    
    paths = save_sim_files() # Create folders to save files
    
    num_of_simulations = 2
    for sim_num in range(num_of_simulations):
        identifier = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        sys = system_wide_sim(travel_matrix,distance_matrix,time_matrix)
        sys.to_excel(os.path.join(paths[0],f"{identifier} Initial-Route-Conditions.xlsx"),index=False)
        sim = sim_by_time(sys,travel_matrix,distance_matrix,time_matrix)
        sim[0].to_excel(os.path.join(paths[1],f"{identifier} simulation.xlsx"),index=False) 
        sim[1].to_excel(os.path.join(paths[2],f"{identifier} tester_updated.xlsx"),index=False) 
        sim[2].to_excel(os.path.join(paths[3],f"{identifier} station_consumption.xlsx"),index=False)
        with pd.ExcelWriter(os.path.join(paths[4],f"{identifier} station_reliability.xlsx")) as writer:
            sim[3].to_excel(writer,sheet_name='Sheet1',index=False) 
            sim[4].to_excel(writer,sheet_name='Sheet2',index=False) 

    # summary_report(paths[1:])
    
'''
issues:
>> ottawa to montreal bck to ontario requires at least 300km range -> start seraching for fuel stations at 45% capacity and refuel near 10%
>> interp station finding need to limit the locations on the way -> distance to interp < distance to dest
    >> if barrie is the destination -> omit port hope for eastbound and cambridge for westbound traffic
>>Traffic from Barrie to Toronto tried to take Tilbury as intermitten
>>Expected issue with Toronto -> Cambridge route since no OnRoute in between
>>Expected issue with Barrie -> Cambridge since no OnRoute in between
>>Expected issue with Detroit -> Windsor route since no OnRoute in between
'''