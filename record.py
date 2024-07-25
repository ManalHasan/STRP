import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Input, BatchNormalization, Activation, Dropout, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the DataFrame from the CSV file
df = pd.read_excel('STRP_d.xlsx')
num_columns = df.shape[1]
nested_list = []
for index, row in df.iterrows():
    # Convert string representations of lists back to lists
    sub_list = []
    for i in range(num_columns - 1):
        sublist1 = eval(row[i])
        sub_list.append(sublist1)
    i_head = eval(row[num_columns - 1])
    sub_list.append(i_head)
    nested_list.append(sub_list)

# Filter out the single-element lists
filtered_tensor = [sample[:-1] for sample in nested_list]#I_head left out for now
filtered_tensor = np.array(filtered_tensor)
 
x_indices = slice(0, 9)  # First 9 features
y_indices = slice(9, 12)  # Next 3 features

# Split the tensor into x and y features
x_features = filtered_tensor[:, :, x_indices]
y_features = filtered_tensor[:, :, y_indices]

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_features, y_features, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Input(shape=x_features.shape[1:]),
    LSTM(64, return_sequences=True),
    LSTM(64, return_sequences=True),
    TimeDistributed(Dense(32, activation='relu')),
    TimeDistributed(Dense(y_features.shape[2]))
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])
#history = model.fit(new_x_features, new_y_features, epochs=150, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])
# Predict
loss = model.evaluate(x_val, y_val)
predictions = model.predict(x_val)

print("Test Loss:", loss)
print("Predictions: ", predictions[0])
print()
print(y_val[0])


# # #########OVERALL P LOSS AND Q LOSS
P_loss=[]
Q_loss=[]
for i in predictions:
    P_loss1=0
    Q_loss1=0
    for j in i:
        P_loss1+=j[1]
        Q_loss1+=j[2]
    # print("Overall P_loss=", P_loss1)
    P_loss+=P_loss1
    print()
    # print("Overall Q_loss=", Q_loss1)
    Q_loss+=Q_loss1

###########calculating R and X using the predicted parameters(p_loss and Q_loss)(and hopefully I_head)
def calc_R_X(P_loss, Q_loss, I_head):
    R_eq=P_loss/(I_head**2)
    X_eq=Q_loss/(I_head**2)
calc_R_X(P_loss[0], Q_loss[0], 50)
#####################################################################################################################################################
# # separate
# ###I_HEAD PREDICTION ####   TRIAL
# new_tensor = [[elem for sublist in sample[:-1] for elem in sublist[:9]] for sample in nested_list]
# for i in range(len(new_tensor)):
#     new_tensor[i].append(nested_list[i][-1][0])
# # print(new_tensor[136])
# new_tensor = np.array(new_tensor)

# new_x_features=new_tensor[:,:-1]
# new_y_features=new_tensor[:,-1]

# # Normalize the features
# # scaler = MinMaxScaler()
# scaler = StandardScaler()
# new_x_features = scaler.fit_transform(new_x_features)
# 8
# # print(max_y , min_y)
# # new_y_features=scalar.

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(new_x_features, new_y_features, test_size=0.2, random_state=42)

# # Build the neural network model
# model = Sequential()
# model.add(Input(shape=(X_train.shape[1],)))
# # model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.01))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1))
# # model = Sequential()
# # model.add(Input(shape=(X_train.shape[1],)))
# # # model.add(Dense(256, activation='relu'))
# # # model.add(BatchNormalization())
# # # model.add(Dropout(0.3))

# # model.add(Dense(128, activation='relu'))
# # model.add(BatchNormalization())
# # model.add(Dropout(0.025))
# # model.add(Dense(64, activation='relu'))
# # model.add(Dropout(0.05))
# # model.add(Dense(32, activation='relu'))
# # model.add(Dense(1))
# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_absolute_error')

# # Define callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
# # model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# # Train the model
# history = model.fit(new_x_features, new_y_features, epochs=150, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# # Evaluate the model
# # model.load_weights('best_model.h5')
# y_pred = model.predict(X_test)
# # mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# # r2 = r2_score(y_test, y_pred)
# # y_pred=y_pred*max_y
# # y_test=
# # print(f'Mean Squared Error: {mse}')
# print(f'Mean Absolute Error: {mae}')
# print("Predictions:", y_pred[:10])
# print()
# print(y_test[:10])
# # # # # print(X_test[0][7],X_test[1][7],X_test[2][7],X_test[3][7],X_test[4][7] )
# # # # # print(f'R^2 Score: {r2}')

# # # # # # option1
# # # # # # # Split the data into training and testing sets
# # # # # # X_train, X_test, y_train, y_test = train_test_split(new_x_features, new_y_features, test_size=0.2, random_state=42)

# # # # # # # Standardize the data (optional but recommended for neural networks)
# # # # # # scaler = StandardScaler()
# # # # # # X_train_scaled = scaler.fit_transform(X_train)
# # # # # # X_test_scaled = scaler.transform(X_test)
# # # # # # X_train_reshaped = np.expand_dims(X_train_scaled, axis=1)
# # # # # # X_test_reshaped = np.expand_dims(X_test_scaled, axis=1)

# # # # # # # Define the model
# # # # # # model = Sequential([
# # # # # #     Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
# # # # #     LSTM(64, return_sequences=True),
# # # # #     LSTM(64, return_sequences=True),
# # # # #     TimeDistributed(Dense(32, activation='relu')),
# # # # #     TimeDistributed(Dense(1))  # Assuming output dimension is 1 (for regression)
# # # # # ])

# # # # # # Compile the model
# # # # # model.compile(optimizer=Adam(learning_rate=0.1), loss='mae', metrics=['mae'])

# # # # # # Train the model
# # # # # history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_data=(X_test_reshaped, y_test))

# # # # # # Predict
# # # # # predictions = model.predict(X_test_reshaped)

# # # # # print("Predictions:", predictions)



# # # # # #OPTION2


# # # # # Split the data into training, validation, and test sets
# # # # X_train, X_temp, y_train, y_temp = train_test_split(new_x_features, new_y_features, test_size=0.3, random_state=42)
# # # # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # # # # Normalize the features
# # # # scaler = StandardScaler()
# # # # X_train_s = scaler.fit_transform(X_train)
# # # # X_val_s = scaler.transform(X_val)
# # # # X_test_s = scaler.transform(X_test)
# # # # # scaler = MinMaxScaler()
# # # # # X_train_s = scaler.fit_transform(X_train)
# # # # # X_val_s = scaler.transform(X_val)
# # # # # X_test_s = scaler.transform(X_test)

# # # # # Define the neural network model
# # # # # model = Sequential()
# # # # # model.add(Input(shape=(new_x_features.shape[1],)))
# # # # # model.add(Dense(64,  activation='relu'))
# # # # # model.add(Dense(32, activation='relu'))
# # # # # model.add(Dense(16, activation='relu'))
# # # # # model.add(Dense(1, activation='linear'))  # Change activation based on your task

# # # # # model = Sequential([
# # # # #     Input(shape=(new_x_features.shape[1],)),
# # # # #     Dense(64, activation='relu'),
# # # # #     BatchNormalization(),
# # # # #     Dropout(0.5),
# # # # #     Dense(32, activation='relu'),
# # # # #     BatchNormalization(),
# # # # #     Dropout(0.5),
# # # # #     Dense(16, activation='relu'),
# # # # #     BatchNormalization(),
# # # # #     Dropout(0.5),
# # # # #     Dense(1, activation='linear')
# # # # # ])

# # # # model = Sequential([
# # # #     Input(shape=(new_x_features.shape[1],)),
# # # #     Dense(512, activation='relu'),
# # # #     BatchNormalization(),
# # # #     Dropout(0.5),
# # # #     Dense(256, activation='relu'),
# # # #     BatchNormalization(),
# # # #     Dropout(0.5),
# # # #     Dense(128, activation='relu'),
# # # #     BatchNormalization(),
# # # #     Dropout(0.5),
# # # #     Dense(64, activation='relu'),
# # # #     BatchNormalization(),
# # # #     Dropout(0.5),
# # # #     Dense(1, activation='relu')
# # # # ])

# # # # # Compile the model
# # # # model.compile(optimizer=Adam(learning_rate=0.01), loss='mse') #0.01 or 0.001  ####0.1 and batchsize=16 ###0.01 nad 32

# # # # # Callbacks for early stopping and model checkpointing
# # # # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # # # # Train the model
# # # # history = model.fit(X_train_s, y_train, epochs=200, validation_data=(X_val_s, y_val),batch_size=32, callbacks=[early_stopping])

# # # # # Evaluate the model on the test set
# # # # loss = model.evaluate(X_test_s, y_test)
# # # # predictions = model.predict(X_test_s)
# # # # print("Test Loss:", loss)
# # # # print("Predictions:", predictions[:10])

# # # # # print(X_test[0][7],X_test[1][7],X_test[2][7],X_test[3][7],X_test[4][7], )
# # # # #######################one more variation
# # # # # model = Sequential([
# # # # #     Input(shape=(new_x_features.shape[1],)),
# # # # #     Dense(512, activation='relu'),
# # # # #     BatchNormalization(),
# # # # #     Dropout(0.5),
# # # # #     Dense(256, activation='relu'),
# # # # #     BatchNormalization(),
# # # # #     Dropout(0.5),
# # # # #     Dense(128, activation='relu'),
# # # # #     BatchNormalization(),
# # # # #     Dropout(0.5),
# # # # #     Dense(128, activation='relu'),
# # # # #     BatchNormalization(),
# # # # #     Dropout(0.5),
# # # # #     Dense(64, activation='relu'),
# # # # #     BatchNormalization(),
# # # # #     Dropout(0.5),
# # # # #     Dense(64, activation='relu'),
# # # # #     BatchNormalization(),
# # # # #     Dropout(0.5),
# # # # #     Dense(1, activation='relu')
# # # # # ])




# # # # ####OPTION 3
# # # # # sequence_length = 10

# # # # # # Ensure enough samples to create sequences
# # # # # num_samples = (new_x_features.shape[0] // sequence_length) * sequence_length
# # # # # new_x_features = new_x_features[:num_samples].reshape((-1, sequence_length, new_x_features.shape[1]))
# # # # # new_y_features = new_y_features[:num_samples].reshape((-1, sequence_length, 1))

# # # # # # Split the data into training, validation, and test sets
# # # # # X_train, X_test, y_train, y_test = train_test_split(new_x_features, new_y_features, test_size=0.2, random_state=42)

# # # # # # Standardize the data
# # # # # scaler = StandardScaler()
# # # # # X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
# # # # # X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# # # # # # # Define the model
# # # # # # model = Sequential([
# # # # # #     Input(shape=(sequence_length, new_x_features.shape[2])),
# # # # # #     LSTM(64, return_sequences=True),
# # # # # #     LSTM(64, return_sequences=True),
# # # # # #     TimeDistributed(Dense(32, activation='relu')),
# # # # # #     TimeDistributed(Dense(1))  # Assuming output dimension is 1 (for regression)
# # # # # # ])

# # # # # # # Compile the model
# # # # # # model.compile(optimizer=Adam(learning_rate=0.1), loss='mae', metrics=['mae'])

# # # # # # # Callbacks for early stopping and learning rate reduction
# # # # # # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# # # # # # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# # # # # # # Train the model
# # # # # # history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping, reduce_lr])

# # # # # # # Predict
# # # # # # predictions = model.predict(X_test_scaled)
# # # # # # print("Test Loss:", model.evaluate(X_test_scaled, y_test))
# # # # # # print("Predictions:", predictions[:10])
# # # # # print(X_test[0])
