import numpy as np
import pandas as pd
import os

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

    # -------------------------Summary 1 - trucks ----------------------
    truck_dist = []
    truck_time = []
    truck_count_refueling_stops = []

    for file_num,file in enumerate(os.listdir(paths[0]),start=1): # Lists all files in the folder and iterates over all
        if file.endswith('.xlsx'): # Check if file is excel file
            file_path = os.path.join(paths[0], file)
            df = pd.read_excel(file_path)
            truck_time.append(df.loc[:149,"Travel Time"].values)
            truck_dist.append(df.loc[:149,"Travel Range"].values)
            truck_count_refueling_stops.append(df.loc[:151,"Refuelling Stops"].values)
    num_simulations = file_num
    truck_dist = [element for row in truck_dist for element in row] # Flatten
    truck_time = [element for row in truck_time for element in row]
    truck_count_refueling_stops = [element for row in truck_count_refueling_stops for element in row]
    max_dist, min_dist, avg_dist = max(truck_dist), min(truck_dist), sum(truck_dist)/len(truck_dist)
    max_time, min_time, avg_time = max(truck_time)/3600, min(truck_time)/3600, sum(truck_time)/(len(truck_time)*3600)
    avg_stops = sum(truck_count_refueling_stops)/len(truck_count_refueling_stops)
    avg_daily_hyd_consump = (avg_dist*50/750)*150 # 50 kg gives 750km, 150 vehicles

    row_labels = ['Average distance travelled (km)','Max. distance (km)','Min. distance (km)','Average travel time (h)','Max. travel time (h)','Min. travel time (h)','Average number of refuelling stops','Max. number of refuelling stops','Avg. daily hydrogen consumption (150 trucks, kg)','Number of simulations run']
    row_vals = [round(avg_dist), round(max_dist), round(min_dist), round(avg_time, 1), round(max_time, 1), round(min_time, 1), round(avg_stops), round(max(truck_count_refueling_stops)),round(avg_daily_hyd_consump,1),num_simulations]
    pd.DataFrame({'Parameter': row_labels, 'Value': row_vals}).to_excel('Truck-Travel-Summary.xlsx',index=False)
    
    # ------------------------------------------Summary 2 fuel station consumption -----------------------------------------
    # Rows = fuel stations
    # Total
    # Cols = average consumption(last row avg across all files), max consumption(across all files), min consumption(across all files), reliability
    num_fuel_stations = 12
    df_dict={}
    for i in range(num_fuel_stations):
        data = {'Maximum consumption': np.zeros(num_simulations),
                'Minimum consumption': np.zeros(num_simulations),
                'Total consumption': np.zeros(num_simulations),
                'Trucks Visited': np.zeros(num_simulations),
                'Trucks Refuelled': np.zeros(num_simulations)}
        
        stat_summary_df = pd.DataFrame(data)
        df_dict[f'Station {i+1}'] = stat_summary_df

    for file_num,file in enumerate(os.listdir(paths[1])): # Lists all files in the folder and iterates over all
        if file.endswith('.xlsx'): # Check if file is excel file
            file_path = os.path.join(paths[1], file)
            df = pd.read_excel(file_path)
            for col_name in df.columns[1:]:
                df_dict[col_name].at[file_num,'Maximum consumption'] = df[col_name].max()
                df_dict[col_name].at[file_num,'Minimum consumption'] = df[col_name].min()
                df_dict[col_name].at[file_num,'Total consumption'] = df[col_name].sum()
    
    for file_num,file in enumerate(os.listdir(paths[2])): # Lists all files in the folder and iterates over all
        if file.endswith('.xlsx'): # Check if file is excel file
            file_path = os.path.join(paths[2], file)

            df = pd.read_excel(file_path,sheet_name=['Sheet1', 'Sheet2'])
            for col_name in df['Sheet1'].columns[1:]:
                df_dict[col_name].at[file_num,'Trucks Visited'] = df['Sheet1'][col_name].sum()
                df_dict[col_name].at[file_num,'Trucks Refuelled'] = df['Sheet2'][col_name].sum()
    
    min_cons = []
    max_cons = []
    avg_total_cons = []
    avg_truck_visit = []
    total_truck_visit = []
    avg_truck_refuel = []
    total_truck_refuel = []
    avg_reliability = []
    row_labels = [f'Station {i+1}' for i in range(num_fuel_stations)]
    for index,val in enumerate(row_labels):
        max_cons.append(round(df_dict[val]['Total consumption']).max())
        min_cons.append(round(df_dict[val]['Total consumption']).min())
        avg_total_cons.append(round(df_dict[val]['Total consumption'].mean()))
        station_visit_avg = df_dict[val]['Trucks Visited'].mean()
        station_refuel_avg = df_dict[val]['Trucks Refuelled'].mean()
        avg_truck_visit.append(round(station_visit_avg))
        avg_truck_refuel.append(round(station_refuel_avg))

        total_truck_visit.append(df_dict[val]['Trucks Visited'].sum())
        total_truck_refuel.append(df_dict[val]['Trucks Refuelled'].sum())
        if station_visit_avg == 0:
            avg_reliability.append(100)
        else:
            avg_reliability.append(round(total_truck_refuel[index]/total_truck_visit[index],5)*100)

    pd.DataFrame({'Station': row_labels, 'Maximum consumption': max_cons,'Minimum consumption':min_cons,'Total consumption':avg_total_cons,'Trucks Visited':avg_truck_visit,'Trucks Refuelled':avg_truck_refuel,'Total Trucks Visited':total_truck_visit,'Total Trucks Refuelled':total_truck_refuel,'Station Reliability':avg_reliability}).to_excel('Fuel-Station-Summary.xlsx',index=False)

    return

if __name__ == "__main__":
    paths = save_sim_files() # Create folders to save files
    summary_report(paths[2:])