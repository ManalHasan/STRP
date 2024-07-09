import py_dss_interface
import os
import pathlib
import pandas as pd
import numpy as np
import csv
import random

class OPENDSS:
    def __init__(self,file_path):
        self.dss = py_dss_interface.DSSDLL()
        self.dss.text(f"compile [{file_path}]")

    def identify_loads(self):
        loads_list=self.dss.loads_all_names()
        return loads_list
        # print(loads_list)

    def identify_line_sections(self):
        lines_list=self.dss.lines_all_names()
        # print (lines_list)
        return lines_list

    def identify_PV(self):
        PV_list=self.dss.pvsystems_all_names()
        #print(PV_list)
        return PV_list

    def identify_battery(self):
        all_elements = self.dss.circuit_all_element_names()
        storage_elements = [el for el in all_elements if el.startswith("Storage.")]
        storage_names = [el.split('.')[1] for el in storage_elements]
        return storage_names
    
    def identify_transformers(self):
        transformers_list = self.dss.transformers_all_Names()
        return transformers_list
    
    def identify_generators(self):
        generators_list = self.dss.generators_all_names()
        return generators_list

    def extract_load_data(self, load_name):
        results = {}
        ans = []
        self.dss.loads_first()
        self.dss.circuit_set_active_element(f"load.{load_name}")
        voltages = self.dss.cktelement_voltages_mag_ang()
        currents = self.dss.cktelement_currents_mag_ang()  # Currents (magnitude and angle)
        # ans += self.dss.loads_read_zipv()
        zip=self.dss.loads_read_zipv()
        
        # print(self.dss.cktelement_currents()) 
        ## per unit voltage
        bus_name = self.dss.cktelement_read_bus_names()[0]
        self.dss.circuit_set_active_bus(bus_name)
        base_voltage = self.dss.bus_kv_base() * 1000

        voltages_magnitudes = voltages[::2]  # Extract magnitude (every other value starting from 0)
        vpu = [v / base_voltage for v in voltages_magnitudes]
        #####
        self.dss.loads_write_name(str(load_name))
        # ans.append(self.dss.loads_read_kw())  # append the float value to the list
        # ans.append(self.dss.loads_read_kvar())  # append the float value to the list
        # ans.append(vpu[0])
        results = {
            'ZIP':zip,
            'kW':self.dss.loads_read_kw(),
            'kvar':self.dss.loads_read_kvar(),
            'Voltages': voltages,
            'Voltages_per_unit': vpu,
            'Currents': currents
        }

        return results

    def extract_line_section_data(self, line_name):
        results={}
        self.dss.lines_first()
        self.dss.circuit_set_active_element(f"line.{line_name}")
        bus1 = self.dss.cktelement_read_bus_names()[0]
        bus2 = self.dss.cktelement_read_bus_names()[1] 
        voltages=self.dss.cktelement_voltages_mag_ang() 
        currents = self.dss.cktelement_currents_mag_ang() 
        powers=self.dss.cktelement_powers()
        # Separate sending and receiving end data
        voltages_bus1 = voltages[:6]  # Assuming 3 phases, first 6 values are for bus1 (magnitude and angle)
        voltages_bus2 = voltages[6:]  # Remaining values are for bus2 (magnitude and angle)
        currents_bus1 = currents[:6]  # First 6 values are for bus1 (magnitude and angle)
        currents_bus2 = currents[6:]  # Remaining values are for bus2 (magnitude and angle)
        powers_bus1 = powers[:2]      # First 2 values are for bus1 (active and reactive power)
        powers_bus2 = powers[6:8]       # Remaining values are for bus2 (active and reactive power) #2:

         # Calculate per-unit voltages for both buses
        self.dss.circuit_set_active_bus(bus1)
        base_voltage_bus1 = self.dss.bus_kv_base() * 1000  # Convert to volts
        voltages_bus1_magnitudes = voltages_bus1[0::2]  # Extract magnitude (every other value starting from 0)
        voltage_per_unit_bus1 = [v / base_voltage_bus1 for v in voltages_bus1_magnitudes]

        self.dss.circuit_set_active_bus(bus2)
        base_voltage_bus2 = self.dss.bus_kv_base() * 1000  # Convert to volts
        voltages_bus2_magnitudes = voltages_bus2[0::2]  # Extract magnitude (every other value starting from 0)
        voltage_per_unit_bus2 = [v / base_voltage_bus2 for v in voltages_bus2_magnitudes]

        results = {
            'Bus1': {
                'Name': bus1,
                'Voltages': voltages_bus1,
                'Voltage_per_unit': voltage_per_unit_bus1,
                'Currents': currents_bus1,
                'Powers': powers_bus1
            },
            'Bus2': {
                'Name': bus2,
                'Voltages': voltages_bus2,
                'Voltage_per_unit': voltage_per_unit_bus2,
                'Currents': currents_bus2,
                'Powers': powers_bus2
            }
        }
        
        return results

    def extract_PV_data(self,pv):
        results={}
        self.dss.pvsystems_first()
        self.dss.circuit_set_active_element(f"pvsystem.{pv}")
        bus = self.dss.cktelement_read_bus_names()[0]
        voltages=self.dss.cktelement_voltages_mag_ang() 
        currents = self.dss.cktelement_currents_mag_ang() 
        powers=self.dss.cktelement_powers()

        ##per unit voltage
        self.dss.circuit_set_active_bus(bus)
        base_voltage = self.dss.bus_kv_base() *1000

        voltages_magnitudes = voltages[::2]  # Extract magnitude (every other value starting from 0)
        vpu = [v / base_voltage for v in voltages_magnitudes]
        #####

        results[pv] = {
            'Voltages': voltages,
            'Voltages_per_unit':vpu,
            'Currents': currents,
            'Power': powers
            }
        return results

    def extract_storage_data(self,battery):
        results={}
        self.dss.pvsystems_first()
        self.dss.circuit_set_active_element(f"Storage.{battery}")
        bus = self.dss.cktelement_read_bus_names()[0]
        voltages=self.dss.cktelement_voltages_mag_ang() 
        currents = self.dss.cktelement_currents_mag_ang() 
        powers=self.dss.cktelement_powers()

        ##per unit voltage
        self.dss.circuit_set_active_bus(bus)
        base_voltage = self.dss.bus_kv_base() *1000

        voltages_magnitudes = voltages[::2]  # Extract magnitude (every other value starting from 0)
        vpu = [v / base_voltage for v in voltages_magnitudes]
        #####

        results[battery] = {
            'Voltages': voltages,
            'Voltages_per_unit':vpu,
            'Currents': currents,
            'Powers': powers
            }
        return results

    def extract_transformer_data(self,transformer):
        results={}
        self.dss.lines_first()
        self.dss.circuit_set_active_element(f"transformer.{transformer}")
        bus1 = self.dss.cktelement_read_bus_names()[0]
        bus2 = self.dss.cktelement_read_bus_names()[1]
        voltages=self.dss.cktelement_voltages_mag_ang() 
        currents = self.dss.cktelement_currents_mag_ang() 
        powers=self.dss.cktelement_powers()
        # Separate sending and receiving end data
        voltages_bus1 = voltages[:6]  # Assuming 3 phases, first 6 values are for bus1 (magnitude and angle)
        voltages_bus2 = voltages[6:]  # Remaining values are for bus2 (magnitude and angle)
        currents_bus1 = currents[:6]  # First 6 values are for bus1 (magnitude and angle)
        currents_bus2 = currents[6:]  # Remaining values are for bus2 (magnitude and angle)
        powers_bus1 = powers[:2]      # First 2 values are for bus1 (active and reactive power)
        powers_bus2 = powers[2:]      # Remaining values are for bus2 (active and reactive power)

         # Calculate per-unit voltages for both buses
        self.dss.circuit_set_active_bus(bus1)
        base_voltage_bus1 = self.dss.bus_kv_base() * 1000  # Convert to volts
        voltages_bus1_magnitudes = voltages_bus1[0::2]  # Extract magnitude (every other value starting from 0)
        voltage_per_unit_bus1 = [v / base_voltage_bus1 for v in voltages_bus1_magnitudes]

        self.dss.circuit_set_active_bus(bus2)
        base_voltage_bus2 = self.dss.bus_kv_base() * 1000  # Convert to volts
        voltages_bus2_magnitudes = voltages_bus2[0::2]  # Extract magnitude (every other value starting from 0)
        voltage_per_unit_bus2 = [v / base_voltage_bus2 for v in voltages_bus2_magnitudes]

        results = {
            'Bus1': {
                'Name': bus1,
                'Voltages': voltages_bus1,
                'Voltage_per_unit': voltage_per_unit_bus1,
                'Currents': currents_bus1,
                'Powers': powers_bus1
            },
            'Bus2': {
                'Name': bus2,
                'Voltages': voltages_bus2,
                'Voltage_per_unit': voltage_per_unit_bus2,
                'Currents': currents_bus2,
                'Powers': powers_bus2
            }
        }
        return results
    

    def extract_gen_data(self,generator):
        results={}
        self.dss.pvsystems_first()
        self.dss.circuit_set_active_element(f"generator.{generator}")
        bus = self.dss.cktelement_read_bus_names()[0]
        voltages=self.dss.cktelement_voltages_mag_ang() 
        currents = self.dss.cktelement_currents_mag_ang() 
        powers=self.dss.cktelement_powers()

        ##per unit voltage
        self.dss.circuit_set_active_bus(bus)
        base_voltage = self.dss.bus_kv_base() *1000

        voltages_magnitudes = voltages[::2]  # Extract magnitude (every other value starting from 0)
        vpu = [v / base_voltage for v in voltages_magnitudes]
        #####

        results[generator] = {
            'Voltages': voltages,
            'Voltages_per_unit':vpu,
            'Currents': currents,
            'Power' : powers
            }
        return results

    def generate_unique_random(self, existing_values, base_value, percentage):
        low = base_value * (1 - percentage)
        high = base_value * (1 + percentage)
        while True:
            value = random.uniform(low, high)
            if value not in existing_values:
                existing_values.add(value)
                return value
            
    def modify_load(self, load_name, new_kw, new_kvar):
        self.dss.circuit_set_active_element(f"load.{load_name}")
        # print("Before modification:", self.dss.cktelement_voltages_mag_ang())
        # print()

        command = f"Edit Load.{load_name} kW={new_kw} kvar={new_kvar}"
        self.dss.text(command)
        self.dss.solution_solve()  # Run the power flow analysis after modifying the load

        # print(f"Modified {load_name}: kW={new_kw}, kvar={new_kvar}")
        self.dss.circuit_set_active_element(f"load.{load_name}")
        # print("After modification:", self.dss.cktelement_voltages_mag_ang())
        # print()


    def dataset_build(self, branch, lines_branch):
        ans = []
        used_kw_values = set()
        used_kvar_values = set()
        percentage=0.2

        # Extract and store original load data
        original_load_data = {}
        for j in branch:
            original_data = self.extract_load_data(j)
            original_kw = original_data['kW']
            original_kvar = original_data['kvar']
            original_load_data[j] = (original_kw, original_kvar)

        for i in range(5):
            single_unit = []
            line_data=self.extract_line_section_data(lines_branch[0])
            for j in branch:
                ##generate random values umique of kw and kvar each time
                # np.random.seed(2)
                original_kw, original_kvar = original_load_data[j]
                # Generate random values within 80-120% of the original values
                new_kw = self.generate_unique_random(used_kw_values, original_kw, percentage)
                new_kvar = self.generate_unique_random(used_kvar_values, original_kvar, percentage)
                
                # Modify the load
                self.modify_load(j, new_kw, new_kvar)

                load_data = self.extract_load_data(j)
                single_load=load_data['ZIP']
                single_load.append(load_data['kW'])
                single_load.append(load_data['kvar'])
                single_load.append(load_data['Voltages_per_unit'][0])
                single_load.append(line_data['Bus1']['Powers'][0]+line_data['Bus2']['Powers'][0])
                single_load.append(line_data['Bus1']['Powers'][1]+line_data['Bus2']['Powers'][1])
                single_unit.append(single_load)
            single_unit.append([line_data['Bus1']['Currents'][0]] )   
            ans.append(single_unit)
        print(ans)
        data_rows = []
        for entry in ans:
            row = entry[0] + entry[1] + entry[2]  # Combine the nested data with y features
            data_rows.append(row)

        # Define column names
        columns = [f'feature_{i+1}' for i in range(len(data_rows[0])-1)] + ['I_head']

        # Create the DataFrame
        df = pd.DataFrame(data_rows, columns=columns)
        # data_array=np.array(ans)
        # num_samples, num_timesteps, num_features = data_array.shape#2 4 8 
        # data_flattened = data_array.reshape(num_samples, -1)
        # # Create column names
        # columns = [f't{t}_f{f}' for t in range(num_timesteps) for f in range(num_features)]

        # # Create the DataFrame
        # df = pd.DataFrame(data_flattened, columns=columns)
        df.to_csv('STRP_data1.csv', index=False)
        # print(df)


if __name__=="__main__":
    script_path=os.path.dirname(os.path.abspath(__file__))
    dss_file=pathlib.Path(script_path).joinpath("Loads (1).dss")
    file_obj= OPENDSS(dss_file)
    # all_loads=file_obj.identify_loads()
    # all_line_sections=file_obj.identify_line_sections()
    # all_PV=file_obj.identify_PV()
    # all_tranformer=file_obj.identify_transformers()
    # all_batteries=file_obj.identify_battery()
    # all_gen=file_obj.identify_generators()
    # data=file_obj.extract_load_data(all_loads[0])
    # # file_obj.write_csv("load_data.csv", data)
    # file_obj.extract_line_section_data(all_line_sections[0])
    # file_obj.extract_PV_data(all_PV[0])
    # file_obj.extract_transformer_data(all_tranformer[0])
    # file_obj.extract_gen_data(all_gen[0])
    # file_obj.extract_storage_data(all_batteries[0])
    # file_obj.modify_load( all_loads[0], 60, 70)
    # file_obj.dataset_build(['2','3', '4', '5'],['1-2','2-3','3-4','4-5'])
    file_obj.dataset_build(['7','8','9','10','11','12','13','14','15','16','17','18'],['6-7','7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15','15-16','16-17','17-18'])


