import py_dss_interface
import os
import pathlib
import pandas as pd
import numpy as np
import random
import re
# import tensorflow as tf

class OPENDSS:
    def __init__(self,file_path):
        self.dss = py_dss_interface.DSSDLL()
        # self.dss = py_dss_interface.DSS()
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
    
    def identify_xy(self):
        xy_list = self.dss.xy

    def plot_correlation_heatmap(self, data):
        # df = pd.DataFrame(data, columns=['Load Name', 'Zp', 'Zi', 'Pc', 'Iu', 'Ip', 'Ic', 'Vm', 'kW', 'kVar', 'Voltage', 'Current'])
        # correlation_matrix = df[['Zp', 'Zi', 'Pc', 'Iu', 'Ip', 'Ic', 'Vm', 'kW', 'kVar', 'Voltage', 'Current']].corr()
        numeric_data = data.drop(columns=['Load Name'])
        # plt.figure(figsize=(12, 8))
        # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        # plt.title('Correlation Heatmap')
        x = data.corr()
        print(x)
        # plt.show()

    def extract_load_data(self,load_name):
        results = {}
        self.dss.loads_first()
        # self.dss.loads_first()
        self.dss.circuit_set_active_element(f"load.{load_name}")
        voltages=self.dss.cktelement_voltages_mag_ang()

        ##per unit voltage
        bus_name=self.dss.cktelement_read_bus_names()[0]
        self.dss.circuit_set_active_bus(bus_name)
        base_voltage = self.dss.bus_kv_base() *1000

        voltages_magnitudes = voltages[::2]  # Extract magnitude (every other value starting from 0)
        vpu = [v / base_voltage for v in voltages_magnitudes]
        vpuu=self.dss.bus_pu_vmag_angle()
        #####
        self.dss.loads_write_name(str(load_name))
        results = {
            'ZIP': self.dss.loads_read_zipv(),
            'kW':self.dss.loads_read_kw(),
            'kVar':self.dss.loads_read_kvar(),
            'kV':self.dss.loads_read_kv(),
            'Voltages': voltages,
            'Voltages per unit':vpu,
            'Currents': self.dss.cktelement_currents_mag_ang()
            }
        
        # print("Load Data:")
        # for k, l in results.items():
        #     print(f"{k}: {l}")
        # print()
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
        powers_bus2 = powers[6:8]      # Remaining values are for bus2 (active and reactive power)

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
                'Bus Name': bus1,
                'Voltages': voltages_bus1,
                'Voltage per unit': voltage_per_unit_bus1,
                'Currents': currents_bus1,
                'Powers': powers_bus1
            },
            'Bus2': {
                'Bus Name': bus2,
                'Voltages': voltages_bus2,
                'Voltage per unit': voltage_per_unit_bus2,
                'Currents': currents_bus2,
                'Powers': powers_bus2
            }
        }
        # print("Bus Data:\n Line Name: ", line_name)
        # for i, j in results.items():
        #     for k, l in j.items():
        #         print(f"{k}: {l}")
        # print()
        return results

    def extract_PV_data(self,pv):
        results={}
        self.dss.pvsystems_first()
        # self.dss.pvsystems_write_name(f"pvsystem.{pv}")
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

        results = {
            'kW': self.dss.pvsystems_kw(),
            'kVar': self.dss.pvsystems_read_kvar(),
            'Voltages': voltages,
            'Voltages per unit':vpu,
            'Currents': currents,
            'Powers': powers,
            'Power Factor': self.dss.pvsystems_read_pf(),
            'Irradiance': self.dss.pvsystems_read_irradiance(),
            'Pmpp': self.dss.pvsystems_read_pmpp(),
            'KVA rating': self.dss.pvsystems_read_kva_rated()
            }
        # print("PV Data")
        # for i, j in results.items():
        #     print(f"{i}: {j}")
        # print()
        return results

    def extract_storage_data(self, battery):
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
        try:
            with open("Branch6-18 3.dss", 'r') as file:
                lines = file.readlines()
        except Exception as e:
            print(f"Error reading the file: {e}")
            return results

        battery_pattern = re.compile(rf'New Storage\.{battery}\b.*', re.IGNORECASE)
        battery_lines = []
        capture = False

        for line in lines:
            if capture:
                if line.strip().startswith('New '):
                    break
                battery_lines.append(line.strip())
            elif battery_pattern.match(line.strip()):
                battery_lines.append(line.strip())
                capture = True

        if not battery_lines:
            print(f"Battery {battery} not found in the OpenDSS file.")
            return results

        battery_definition = ' '.join(battery_lines)
        
        # print(f"Extracted battery definition: {battery_definition}")

        stored_percentage_match = re.search(r'%stored\s*=\s*(\d+\.?\d*)', battery_definition)
        reserved_percentage_match = re.search(r'%reserve\s*=\s*(\d+\.?\d*)', battery_definition)
        kw_output_match = re.search(r'kWrated\s*=\s*(\d+\.?\d*)', battery_definition)
        kwh_match = re.search(r'kWHrated\s*=\s*(\d+\.?\d*)', battery_definition)
        state_match = re.search(r'state\s*=\s*(\w+)', battery_definition)
        eff_match = re.search(r'%EffCharge\s*=\s*(\d+\.?\d*)', battery_definition) 
        kvar_match = re.search(r'kvar\s*=\s*(-?\d+\.?\d*)', battery_definition)
        # print(kvar_match)
        stored_percentage = float(stored_percentage_match.group(1)) if stored_percentage_match else None
        kw_output = float(kw_output_match.group(1)) if kw_output_match else None
        kwh = float(kwh_match.group(1)) if kwh_match else None
        kvar = (kvar_match.group(1)) if kvar_match else None
        state = state_match.group(1) if state_match else None
        reserved_percentage = float(reserved_percentage_match.group(1)) if reserved_percentage_match else None
        eff_charge = float(eff_match.group(1)) if eff_match else None

        results = {
            # 'Voltages': voltages,
            'Voltages per unit':vpu,
            # 'Currents': currents,
            'Powers': powers,
            'Stored %': stored_percentage,
            'Reserved %': reserved_percentage,
            'Eff Charge': eff_charge,
            'kW Output': kw_output,
            'State': 0 if state == 'discharge' else 1,
            'kWH': kwh,
            'kVar': kvar
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
        powers_bus2 = powers[6:8]      # Remaining values are for bus2 (active and reactive power)

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
                'Voltages per unit': voltage_per_unit_bus1,
                'Currents': currents_bus1,
                'Powers': powers_bus1
            },
            'Bus2': {
                'Name': bus2,
                'Voltages': voltages_bus2,
                'Voltages per unit': voltage_per_unit_bus2,
                'Currents': currents_bus2,
                'Powers': powers_bus2
            }
        }
        print("Transformer Data")
        for i, j in results.items():
            for k, l in j.items():
                print(f"{k}: {l}")
        print()
        return results
    

    def extract_gen_data(self,generator):
        results={}
        self.dss.pvsystems_first()
        # self.dss.generators_write_name(f"generator.{generator}")
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
            'Voltages per unit':vpu,
            'Currents': currents,
            'Power' : powers,
            'Power Factor': self.dss.generators_read_kva_rated()
            }
        print("Generator Data")
        for i, j in results.items():
            for k, l in j.items():
                print(f"{k}: {l}")
        print()
        return results
    
    # def extract_XYCurve(self, )
    
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
        command2 = f"Edit Load.{load_name} kW={new_kw} kvar={new_kvar} ZIPV={new_zipv}"
        self.dss.text(command2)
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
    
    def modify_pv(self, pv, new_irrad):
        self.dss.circuit_set_active_element(f"PVSystem.{pv}")
        command1 = f"Edit PVSystem.{pv}  irradiance={new_irrad}"
        self.dss.text(command1)
        self.dss.circuit_set_active_element(f"PVSystem.{pv}")
        self.dss.solution_solve()

    def modify_battery(self, battery, new_state, new_stored, new_bkW):
        self.dss.circuit_set_active_element(f"Storage.{battery}")
        if new_state == 0:
            command = f"Edit Storage.{battery}  %stored={new_stored} state=discharge kWrated={new_bkW}"
        else:
            command = f"Edit Storage.{battery}  %stored={new_stored} state=charge kWrated={new_bkW}"
        self.dss.text(command)
        self.dss.solution_solve()
        self.dss.circuit_set_active_element(f"Storage.{battery}")

    def dataset_build(self, branch, lines_branch, pv, battery):
        ans = []
        used_kw_values = set()
        used_kvar_values = set()
        percentage=0.5

        # Extract and store original load data
        original_load_data = {}

        original_battery_data = {}
        for b in battery:
            orig_battery = self.extract_storage_data(b)
            original_battery_data[b] = (orig_battery['kW Output'])

        for j in branch:
            original_data = self.extract_load_data(j)
            original_kw = original_data['kW']
            original_kvar = original_data['kVar']
            original_load_data[j] = (original_kw, original_kvar)
    

        for i in range(5000):
            single_unit = []
            line_data=self.extract_line_section_data(lines_branch[0])##I at the head of the branch

            new_irrad = round(np.random.uniform(0, 1), 1)
            self.modify_pv(pv, new_irrad)
            battery_data = []
            new_state1 = random.randint(0, 1)
            new_state2 = 1 - new_state1
            for b, new_state in zip(battery, [new_state1, new_state2]):
                # single_batt = []
                new_stored = round(np.random.uniform(0, 100), 1)
                orig_battery = self.extract_storage_data(b)
                original_b_kW = original_battery_data[b]
                new_bkW = round(np.random.uniform(0, original_b_kW), 1)
                self.modify_battery(b, new_state, new_stored, new_bkW)
                battery_data.append(new_stored)
                battery_data.append(new_state)
                battery_data.append(new_bkW) 
            single_unit.append(battery_data)
            
            for j in branch:
                original_kw, original_kvar = original_load_data[j]
                # Generate random values within 80-120% of the original values
                new_kw = self.generate_unique_random(used_kw_values, original_kw, percentage)
                new_kvar = self.generate_unique_random(used_kvar_values, original_kvar, percentage)
                new_zipv = self.modify_zip()

                # Modify the load
                self.modify_load(j, new_kw, new_kvar, new_zipv)

                load_data = self.extract_load_data(j)
                line_data1=self.extract_line_section_data(lines_branch[branch.index(j)])
                single_load=load_data['ZIP']
                single_load.append(load_data['kW'])
                single_load.append(load_data['kVar'])
                single_load.append(load_data['Voltages per unit'][0])
                single_load.append(line_data1['Bus1']['Powers'][0]+line_data1['Bus2']['Powers'][0])
                single_load.append(line_data1['Bus1']['Powers'][1]+line_data1['Bus2']['Powers'][1])
                single_unit.append(single_load)
            I_head = [line_data['Bus1']['Currents'][0]]
            self.dss.circuit_set_active_element(f"PVSystem.{pv}")
            pv_data = self.extract_PV_data(pv)
            I_head.append(pv_data['Irradiance'])
            single_unit.append(I_head)  
            ans.append(single_unit)
        # print(ans)

    # Create the DataFrame
        df = pd.DataFrame(ans)
        df.to_excel('STRP_data4.xlsx', index=False)
        # print(df)0   
    def parameter(self, loads,lines):
        ans=[]
        all_batteries=self.identify_battery()
        line_data=self.extract_line_section_data(lines[0])
        batteries=[]
        for i in all_batteries:
            battery_data=self.extract_storage_data(i)
            batteries.append(battery_data['Stored %'])
            batteries.append(battery_data['State'])
            batteries.append(battery_data['kW Output'])
        ans.append(batteries)
        for j in loads:
            load_data = self.extract_load_data(j)
            single_load=load_data['ZIP']
            single_load.append(load_data['kW'])
            single_load.append(load_data['kVar'])
            single_load.append(load_data['kV'])
            single_load.append(load_data['Voltages per unit'][0])
            ans.append(single_load)
        I_head = [line_data['Bus1']['Currents'][0]]
        self.dss.circuit_set_active_element(f"PVSystem.{self.identify_PV()[0]}")
        pv_data = self.extract_PV_data(self.identify_PV()[0])
        I_head.append(pv_data['Irradiance'])
        ans.append(I_head)
        # print(ans)  
        return ans

if __name__=="__main__":
    script_path=os.path.dirname(os.path.abspath(__file__))
    dss_file = pathlib.Path(script_path).joinpath("Branch6-18 3.dss")
    file_obj= OPENDSS(dss_file)
    #file_obj.parameter(['7','8','9','10','11','12','13','14','15','16','17','18'],['6-7','7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15','15-16','16-17','17-18'])
    all_PV=file_obj.identify_PV()
    all_batteries=file_obj.identify_battery()
    file_obj.dataset_build(['7','8','9','10','11','12','13','14','15','16','17','18'],['6-7','7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15','15-16','16-17','17-18'], all_PV[0], all_batteries)


