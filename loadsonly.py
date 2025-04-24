import py_dss_interface
import os
import pathlib
import pandas as pd
import random
import numpy as np

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
        ## per unit voltage
        bus_name = self.dss.cktelement_read_bus_names()[0]
        self.dss.circuit_set_active_bus(bus_name)
        base_voltage = self.dss.bus_kv_base() * 1000

        voltages_magnitudes = voltages[::2]  # Extract magnitude (every other value starting from 0)
        vpu = [v / base_voltage for v in voltages_magnitudes]
        #####
        self.dss.loads_write_name(str(load_name))
        zip=self.dss.loads_read_zipv()

        results = {
            'ZIP':zip,
            'kW':self.dss.loads_read_kw(),
            'kvar':self.dss.loads_read_kvar(),
            'kV':self.dss.loads_read_kv(),
            # 'Voltages': voltages,
            'Voltages_per_unit': vpu,
            # 'Currents': currents
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
        powers_bus2 = powers[6:8]     # Remaining values are for bus2 (active and reactive power) #2:

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
        powers_bus2 = powers[6:8]     # Remaining values are for bus2 (active and reactive power)

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
            
    def modify_load(self, load_name, new_kw, new_kvar, new_zipv):
        self.dss.circuit_set_active_element(f"load.{load_name}")

        command = f"Edit Load.{load_name} kW={new_kw} kvar={new_kvar} ZIPV={new_zipv}"
        self.dss.text(command)
        self.dss.solution_solve()  # Run the power flow analysis after modifying the load

        # print(f"Modified {load_name}: kW={new_kw}, kvar={new_kvar}")
        self.dss.circuit_set_active_element(f"load.{load_name}")
        # print("After modification:", self.dss.cktelement_voltages_mag_ang())

    def modify_zip(self):
        # Generate random values for the real power coefficients and normalize them to sum to 1
        Zp = round(np.random.uniform(0, 1), 1)
        Zi = round(np.random.uniform(0, abs(1 - Zp)), 1)
        Pc = round(abs(1 - Zp - Zi), 1)
        
        Ip = round(np.random.uniform(0, 1), 1)
        Ic = round(np.random.uniform(0, abs(1 - Ip)), 1)
        Pi = round(abs(1 - Ip - Ic), 1)
        
        Vm = 0
        zipv = [Zp, Zi, (Pc), Ip, Ic, Pi, Vm]
        zipv = [float(x) for x in zipv]
        # print(zipv)
        return zipv

    def dataset_build(self, branch, lines_branch):
        ans = []
        used_kw_values = set()
        used_kvar_values = set()
        percentage=0.5

        # Extract and store original load data
        original_load_data = {}
        for j in branch:
            original_data = self.extract_load_data(j)
            original_kw = original_data['kW']
            original_kvar = original_data['kvar']
            original_load_data[j] = (original_kw, original_kvar)

        for i in range(5000):
            single_unit = []
            line_data=self.extract_line_section_data(lines_branch[0])##I at the head of the branch
            for j in branch:
                original_kw, original_kvar = original_load_data[j]
                # Generate random values within 80-120% of the original values
                new_kw = self.generate_unique_random(used_kw_values, original_kw, percentage)
                new_kvar = self.generate_unique_random(used_kvar_values, original_kvar, percentage)
                new_zipv = self.modify_zip()

                # Modify the load
                self.modify_load(j, new_kw, new_kvar, new_zipv)
                
                line_data1=self.extract_line_section_data(lines_branch[branch.index(j)])
                load_data = self.extract_load_data(j)
                single_load=load_data['ZIP']
                single_load.append(load_data['kW'])
                single_load.append(load_data['kvar'])
                single_load.append(load_data['Voltages_per_unit'][0])
                single_load.append(line_data1['Bus1']['Powers'][0]+line_data1['Bus2']['Powers'][0])
                single_load.append(line_data1['Bus1']['Powers'][1]+line_data1['Bus2']['Powers'][1])
                single_unit.append(single_load)
            single_unit.append([line_data['Bus1']['Currents'][0]])   
            # print([line_data['Bus1']['Currents'][0]])
            ans.append(single_unit)
        # print(ans)

    # Create the DataFrame
        df = pd.DataFrame(ans)
        df.to_excel('STRP_d.xlsx', index=False)
        # print(df)
    def parameters(self,load_list, line_list):
        orignal=[]
        line_data=self.extract_line_section_data(line_list[0])
        for i in load_list:
            original_data = self.extract_load_data(i)
            l=original_data['ZIP']
            l.append(original_data['kW'])
            l.append(original_data['kvar'])
            l.append(original_data['kV'])
            l.append(original_data['Voltages_per_unit'][0])
            #add P_loss
            orignal.append(l)
        orignal.append([line_data['Bus1']['Currents'][0]] )
        return orignal

if __name__=="__main__":
    script_path=os.path.dirname(os.path.abspath(__file__))
    dss_file=pathlib.Path(script_path).joinpath("Onlyloadsfinal1.dss")
    file_obj= OPENDSS(dss_file)
    # print(file_obj.parameters(['7','8','9','10','11','12','13','14','15','16','17','18'],['6-7','7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15','15-16','16-17','17-18']))
    file_obj.dataset_build(['7','8','9','10','11','12','13','14','15','16','17','18'],['6-7','7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15','15-16','16-17','17-18'])
    # file_obj.dataset_build(['7','8', '9'],['6-7','7-8', '8-9']) 
   
