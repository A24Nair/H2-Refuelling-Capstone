import numpy as np
import pandas as pd
import random
import find_dist_excel_matrix as fd

def new_node(ro,tm,dm,tim,stn): #ro is the truck info, tm is the travel matrix, stn is the remaining fuel in stations
    dest_node = np.array([tm.iloc[12:,0]]).flatten().tolist()
    probabilities = np.array([0.1,0.1,0.07,0.05,0.05,0.13,0.3,0.2])
    interp_rnge = np.array([])
    if int(float(ro['Intermittent Stop'])) != 0: #else take the intermitten stop as the starting stop
        fuel_left = float(ro["Remaining Fuel"]) - dm.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Intermittent Stop"]))]/700*60
        ro["Travel Time"] = float(ro["Travel Time"]) + float(ro["Remaining Time"])
        ro["Travel Range"] = float(ro["Travel Range"]) + dm.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Intermittent Stop"]))]
        fuel_needed = 60 - fuel_left
        #print(fuel_needed)
        station = int(float(ro["Intermittent Stop"])) - 1
        stn[station] -= fuel_needed
        ro["Remaining Fuel"] = 60
        ro["Remaining Range"] = 700
        ro['Starting Node'] = int(float(ro['Intermittent Stop']))
        #need to check again for intermitten
        if float(ro["Remaining Range"]) - dm.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))] <= 700*0.45 and (int(float(ro["Starting Node"]))*int(float(ro["Ending Node"]))!=304):
            inter_stat = tm.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))] == tm.iloc[int(float(ro["Starting Node"]))-1] #returns a row or true and false to see which stations are compatible
            col_ind = np.where(inter_stat.values.flatten())[0]
            col_ind = col_ind[col_ind < 13] #filters for the compatable refuelling stations
            for j in col_ind:
                rem_rnge = float(ro["Remaining Range"]) - dm.iloc[int(float(ro["Starting Node"]))-1,j]
                if rem_rnge > 0 and dm.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))] > dm.iloc[int(float(ro["Starting Node"]))-1,j]:
                    if (int(float(ro["Ending Node"])) == 16 and j == 8) or (int(float(ro["Ending Node"])) == 16 and j == 7):
                        interp_rnge = np.append(interp_rnge,9999)
                    else:
                        interp_rnge = np.append(interp_rnge, rem_rnge)
                else:
                    interp_rnge = np.append(interp_rnge,9999)
            if len(interp_rnge[interp_rnge<9000]) != 0:
                try:
                    ro["Intermittent Stop"] = col_ind[(interp_rnge - (700*10)).argmin()] #obtains the intermittent station number
                    ro["Remaining Time"] = tim.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Intermittent Stop"]))]
                except:#seems ottawa to montreal route is having issues
                    #print(ro["Starting Node"],ro["Ending Node"],ro["Remaining Range"])
                    #print("failed to find a suitable station, the range is enough to go from ottawa to montreal and back")
                    ro["Remaining Time"] = tim.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))]
            else:
                ro["Remaining Time"] = tim.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))]
                ro["Intermittent Stop"] = 0
        else:
            ro["Remaining Time"] = tim.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))]
            ro["Intermittent Stop"] = 0
        
    else: #if stop == 0 -> let ending node be the starting node and find a new ending node
        ro["Travel Time"] = float(ro["Travel Time"]) + float(ro["Remaining Time"])
        ro["Travel Range"] = float(ro["Travel Range"]) + dm.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))]
        ro["Remaining Range"] = float(ro["Remaining Range"]) - dm.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))]
        ro["Remaining Fuel"] = float(ro["Remaining Fuel"]) - dm.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))]/700*60
        ro['Starting Node'] = int(float(ro['Ending Node']))
        end_node = np.random.choice(dest_node, p = probabilities)
        time_check = 43200 - float(ro["Travel Time"]) - tim.iloc[int(float(ro["Starting Node"]))-1,end_node]
        counter = 0
        while float(end_node) == float(ro["Starting Node"]) and (counter < 10 or time_check < 0):
            end_node = np.random.choice(dest_node, p = probabilities)
            time_check = 36000 - float(ro["Travel Time"]) - tim.iloc[int(float(ro["Starting Node"]))-1,end_node] #13h max -> 10h right now
            counter += 1
        if counter > 9 or time_check < 0: #end the truck sim if more than 10 fails
            ro["Remaining Time"] = 1000000
            ro["Ending Node"] = "Finished"
            return ro,stn
        ro["Ending Node"] = end_node
        
        if float(ro["Remaining Range"]) - dm.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))] <= 700*0.45 and (int(float(ro["Starting Node"]))*int(float(ro["Ending Node"]))!=304):
            inter_stat = tm.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))] == tm.iloc[int(float(ro["Starting Node"]))-1] #returns a row or true and false to see which stations are compatible
            col_ind = np.where(inter_stat.values.flatten())[0]
            col_ind = col_ind[col_ind < 13] #filters for the compatable refuelling stations
            for j in col_ind:
                rem_rnge = float(ro["Remaining Range"]) - dm.iloc[int(float(ro["Starting Node"]))-1,j]
                if rem_rnge > 0 and dm.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))] > dm.iloc[int(float(ro["Starting Node"]))-1,j]:
                    if (int(float(ro["Ending Node"])) == 16 and j == 8) or (int(float(ro["Ending Node"])) == 16 and j == 7):
                        interp_rnge = np.append(interp_rnge,9999)
                    else:
                        interp_rnge = np.append(interp_rnge, rem_rnge)
                else:
                    interp_rnge = np.append(interp_rnge,9999)
            if len(interp_rnge[interp_rnge<9000]) != 0:
                try:
                    ro["Intermittent Stop"] = col_ind[(interp_rnge - (700*10)).argmin()] #obtains the intermittent station number
                    ro["Remaining Time"] = tim.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Intermittent Stop"]))]
                except:#seems ottawa to montreal route is having issues
                    #print(ro["Starting Node"],ro["Ending Node"],ro["Remaining Range"])
                    #print("failed to find a suitable station, the range is enough to go from ottawa to montreal and back")
                    ro["Remaining Time"] = tim.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))]
            else:
                ro["Remaining Time"] = tim.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))]
                ro["Intermittent Stop"] = 0      
        else:
            ro["Remaining Time"] = tim.iloc[int(float(ro["Starting Node"]))-1,int(float(ro["Ending Node"]))]  
    return ro,stn

def system_wide_sim(tm,dm,tim):
    title = np.array(["Truck #","Starting Node","Ending Node","Intermittent Stop","Remaining Range","Remaining Fuel","Remaining Time","Travel Time","Travel Range"])
    # Rows are number of vehicles = 150 + 6 tube trailers
    # Variables below are columns =  9 columns
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
    trnge = np.zeros((156,1))
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
        #Finding intermitten stations if needed
        if float(rnge[i-1,0]) - dm.iloc[int(start_dest[i-1,0]-1),int(end_dest[i-1,0])] <= 700*0.45 and (start_dest[i-1,0]*end_dest[i-1,0] != 304): #does not allow toronto <> barrie routes to find intermitten stns
            inter_stat = tm.iloc[int(start_dest[i-1,0]-1),int(end_dest[i-1,0])] == tm.iloc[int(start_dest[i-1,0]-1)] #returns a row or true and false to see which stations are compatible
            col_ind = np.where(inter_stat.values.flatten())[0]
            col_ind = col_ind[col_ind < 13] #filters for the compatable refuelling stations
            for j in col_ind:
                rem_rnge = float(rnge[i-1,0]) - dm.iloc[int(start_dest[i-1,0]-1),j]
                if rem_rnge > 0 and dm.iloc[int(start_dest[i-1,0]-1),int(end_dest[i-1,0])] > dm.iloc[int(start_dest[i-1,0]-1),j]:
                    if (end_dest[i-1,0] == 16 and j == 8) or (end_dest[i-1,0] == 16 and j == 7):#if (end_dest[i-1,0] == 16 and j != 8) or (end_dest[i-1,0] == 16 and j != 7) or (end_dest[i-1,0] != 16):
                        interp_rnge = np.append(interp_rnge,9999)
                    else:
                        interp_rnge = np.append(interp_rnge, rem_rnge)
                else:
                    interp_rnge = np.append(interp_rnge,9999)
            if len(interp_rnge[interp_rnge<9000]) != 0:
                try:
                    interp_stn = col_ind[(interp_rnge - (700*10)).argmin()] #obtains the intermittent station number
                    rem_time[i-1,0] = tim.iloc[int(start_dest[i-1,0]-1),interp_stn]
                except:#seems ottawa to montreal route is having issues
                    print(start_dest[i-1,0],end_dest[i-1,0],rnge[i-1,0])
                    print("failed to find a suitable station, the range is enough to go from ottawa to montreal and back")
                    rem_time[i-1,0] = tim.iloc[int(start_dest[i-1,0]-1),int(end_dest[i-1,0])]
            else:
                rem_time[i-1,0] = tim.iloc[int(start_dest[i-1,0]-1),int(end_dest[i-1,0])]  
        else:
            rem_time[i-1,0] = tim.iloc[int(start_dest[i-1,0]-1),int(end_dest[i-1,0])]
        intermitten[i-1] = interp_stn
        interp_stn = 0
        
    start_dest[150,0] = start_dest[151,0] = start_dest[152,0] = 3 #this could lead to trouble later
    start_dest[153,0] = start_dest[154,0] = start_dest[155,0] = 8
    rem_time[150:,0] = 99999
    
    combined_matrix = np.vstack((truck_num,tube_num))
    combined_matrix = np.hstack((combined_matrix,start_dest,end_dest,intermitten,rnge,fuel,rem_time,trav_time,trnge))
    combined_matrix = np.vstack((title,combined_matrix))
    combined_matrix = pd.DataFrame(combined_matrix)
    combined_matrix.columns = combined_matrix.iloc[0]
    combined_matrix = combined_matrix.drop(0)
    
    return combined_matrix

def sim_by_time(sys_all,tm,dm,tim): #takes the entire starting matrix to start simulation
    stat_mtrx = np.arange(1,157)
    plc_hldr = np.array([])
    for i in range (1,13):
        plc_hldr = np.append(plc_hldr, "Station " + str(i))
    stat_mtrx = np.hstack((stat_mtrx,plc_hldr)) #establishes title row of the simulation
    stat_mtrx = np.insert(stat_mtrx, 0, "Time")
    stat_mtrx = pd.DataFrame([stat_mtrx])
    stat_mtrx.columns = stat_mtrx.iloc[0]
    stat_mtrx = stat_mtrx.drop(0)
    init_stn = np.array([1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000])
    times_range = np.arange(0,50401,600).T.reshape(-1,1)
    travel_time = np.array([sys_all["Remaining Time"]])
    old_time = np.array(travel_time.astype(float))
    for i in range(len(times_range)):
        if i == 0:
            ro = np.hstack((i,old_time.flatten(),init_stn))
            stat_mtrx.loc[len(stat_mtrx)] = ro
            continue
        new_time = np.array(old_time - 600)
        time_check = new_time[new_time <= 0]
        if len(time_check) > 0:
            new_loc = np.where(new_time <= 0)[1] #returns a row of array indexes that arrived
            for j in new_loc:
                row_data = sys_all.iloc[j] #go to the jth row of sys_all -> truck info
                row_data,init_stn = new_node(row_data,tm,dm,tim,init_stn)#put into new node function
                sys_all.iloc[j] = row_data
                new_time[0][j] = row_data["Remaining Time"]
        #take output and take the new remaining time into the new_time slot
        ro = np.hstack((i*600,new_time.flatten(),init_stn))
        stat_mtrx.loc[len(stat_mtrx)] = ro
        old_time = new_time
    return stat_mtrx,sys_all
    
if __name__ == "__main__":
    excel_file_path = "Travel Matrix.xlsx" # Enter filepath here
    #df = pd.read_excel(excel_file_path,sheet_name="Addresses",usecols="A:G",nrows=22) # Missing: build logic for last row with non empty postal code
    # Apply the function
    #df[["Latitude", "Longitude"]] = df.apply(find_location, axis=1)
    #print(df)
    #tester = find_travel_time_distance(df.iloc[5,7],df.iloc[5,8],df.iloc[18,7],df.iloc[18,8]) #this function can be used to extract time and distance -> to work on it more
    travel_matrix = pd.read_excel(excel_file_path,sheet_name="Service-Directions",usecols="A:U",nrows=22)
    distance_matrix = pd.read_excel(excel_file_path,sheet_name="Distance-Matrix",usecols="A:U",nrows=22)
    time_matrix = pd.read_excel(excel_file_path,sheet_name="Driving-Time-Matrix",usecols="A:U",nrows=22)
    
    sys = system_wide_sim(travel_matrix,distance_matrix,time_matrix)
    sys.to_excel("tester.xlsx",index=False) #adjust file path here
    sim = sim_by_time(sys,travel_matrix,distance_matrix,time_matrix)
    sim[0].to_excel("simulation.xlsx",index=False) #adjust file path here
    sim[1].to_excel("tester_updated.xlsx",index=False) #adjust file path here
    
'''
remaining items in this code:
    >>Run the code and graph the results on Excel
    >>Tube trailer logic
    >>13h 10 stations multi simulation -> append
'''

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