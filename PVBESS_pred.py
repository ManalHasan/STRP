import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import py_dss_interface
import pathlib
import math
import ast
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import PVBESS
class Prediction:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = None
        self.x_train, self.x_val, self.y_train, self.y_val = self.load_and_prepare_data()

    def load_and_prepare_data(self):
        # Load the DataFrame from the file
        df = pd.read_excel(self.file_path)
        # Initialize an empty list to hold the nested lists
        nested_list = []

        # Iterate over each row in the dataframe
        for index, row in df.iterrows():
            # Initialize a sub-list for the current row
            sub_list = []
            
            # Convert each cell in the row, excluding the last one, to a list
            for i in range(df.shape[1] - 1):
                sub_list.append(ast.literal_eval(row[i]))
            
            # Convert the last cell in the row to a list and append it to the sub-list
            sub_list.append(ast.literal_eval(row[df.shape[1] - 1]))
            
            # Append the sub-list to the main nested list
            nested_list.append(sub_list)

        # Output the nested list to verify
        # print(nested_list[:2])
        filtered=[]
        for sublist in nested_list:
            l=[]
            l1=sublist[0]
            l1.append(sublist[-1][1])
            for el_list in sublist[1:-1]:
                l2=l1
                l2=l2+el_list
                l.append(l2)
            filtered.append(l)
        # print(filtered[0])
        filtered_tensor = np.array(filtered)
        
        x_indices = slice(0, 16)  # First 9 features
        y_indices = slice(16, 19)  # Next 3 features

        # Split the tensor into x and y features
        x_features = filtered_tensor[:, :, x_indices]
        y_features = filtered_tensor[:, :, y_indices]

        # Split the data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_features, y_features, test_size=0.2, random_state=42)
        return x_train, x_val, y_train, y_val

    def train_model(self):
        # Define the model
        self.model = Sequential([
            Input(shape=self.x_train.shape[1:]),
            LSTM(64, return_sequences=True),
            LSTM(64, return_sequences=True),
            TimeDistributed(Dense(32, activation='relu')),
            TimeDistributed(Dense(self.y_train.shape[2]))
        ])

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        self.model.fit(self.x_train, self.y_train, epochs=50, batch_size=32, validation_data=(self.x_val, self.y_val), callbacks=[early_stopping])

    def predict(self, x_features, I_head):
        if self.model is None:
            raise ValueError("Model is not trained. Please train the model first using train_model method.")
        x_features = np.expand_dims(x_features, axis=0)
        loss = self.model.evaluate(self.x_val, self.y_val)
        predictions = self.model.predict(x_features)
        print("Test Loss:", loss)
        return predictions, I_head

    def calculate_p_q_loss(self, predictions):
        # P_loss = []
        # Q_loss = []
        P_loss = 0
        Q_loss = 0
        for i in predictions:
            P_loss += np.sum(i[:, 1])  # Sum all P_loss values in the prediction
            Q_loss += np.sum(i[:, 2])
        return P_loss, Q_loss

    def calc_R_X(self, P_loss, Q_loss, I_head):
        R_eq = P_loss / (I_head**2)
        X_eq = Q_loss / (I_head**2)
        return R_eq, X_eq

    def pred_zipv(self,p, predictions, V_LN_r):
        new_zip=[]
        for i in range( len(p)):
            P_b=p[i][7]*((p[i][0]*(predictions[0][i][0]**2)/(V_LN_r**2)) + (p[i][1]*(predictions[0][i][0])/(V_LN_r)) + p[i][2])#active
            P_b_r=p[i][8]*((p[i][3]*(predictions[0][i][0]**2)/(V_LN_r**2)) + (p[i][4]*(predictions[0][i][0])/(V_LN_r)) + p[i][5])#reactive

            load=[]
            load.append((p[i][7]*p[i][0]*(predictions[0][i][0]**2))/(P_b*(V_LN_r**2)))#az
            load.append((p[i][7]*p[i][1]*predictions[0][i][0])/(P_b*V_LN_r))          #ai
            load.append((p[i][7]*p[i][2])/P_b)                                        #ap

            load.append((p[i][8]*p[i][3]*(predictions[0][i][0]**2))/(P_b_r*(V_LN_r**2)))#rz
            load.append((p[i][8]*p[i][4]*predictions[0][i][0])/(P_b_r*V_LN_r))          #ri
            load.append((p[i][8]*p[i][5])/P_b_r)                                        #rp
            load.append(0.0)
            new_zip.append(load)

        print(new_zip)

if __name__ == "__main__":
    file_path = 'STRP_data4.xlsx'

    # Initialize and train the prediction model
    prediction_model = Prediction(file_path)
    prediction_model.train_model() 

    script_path=os.path.dirname(os.path.abspath(__file__))
    dss_file=pathlib.Path(script_path).joinpath("Branch6-18 3.dss")
    file_obj= PVBESS.OPENDSS(dss_file)
    parameter=file_obj.parameter(['6','7','8','9','10','11','12','13','14','15','16','17','18'],['6-7','7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15','15-16','16-17','17-18'])
    print(parameter)
    l=[]
    l1=parameter[0]#batteries data
    l1.append(parameter[-1][1])#irradiance
    for el_list in parameter[2:-1]:#exclude load 6
        l2=l1
        l2=l2+el_list[:9]#7 zip v, kW, kVar
        l.append(l2)
    # print(l)
    # Make predictions
    predictions, I_head = prediction_model.predict(np.array(l), parameter[-1][0])
    # print(I_head)
    # Calculate P_loss and Q_loss
    P_loss, Q_loss = prediction_model.calculate_p_q_loss(predictions)
    # print(P_loss, Q_loss)
    # Calculate R and X
    R_eq, X_eq = prediction_model.calc_R_X(P_loss, Q_loss, I_head)
    #Calculated+ predicted vpu
    V_LL_s=parameter[1][9]*parameter[1][-1]
    V_LN_s=V_LL_s/math.sqrt(3)
    V_LN_r=V_LN_s-(I_head*(R_eq+X_eq))
    V_LN_r=V_LN_r/(parameter[1][9]/math.sqrt(3))
    
    p=parameter[2:-1]#7-18 (exclude 6)
    prediction_model.pred_zipv(p, predictions,V_LN_r )
    print(predictions[0])
    # print(R_eq, X_eq)