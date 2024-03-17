import numpy as np
import pandas as pd
import os
from datetime import datetime

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
    paths = save_sim_files() # Create folders to save files
    summary_report(paths[1:])