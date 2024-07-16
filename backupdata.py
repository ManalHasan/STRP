import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Input, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the DataFrame from the CSV file
df = pd.read_excel('STRP_data1.xlsx')
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

# # Filter out the single-element lists
# filtered_tensor = [sample[:-1] for sample in nested_list]#I_head left out for now
# filtered_tensor = np.array(filtered_tensor)
# # Check the structure after filtering
# print(f"Filtered tensor shape: {filtered_tensor.shape}")

# # Define indices for splitting into x and y features
# x_indices = slice(0, 9)  # First 9 features
# y_indices = slice(9, 12)  # Next 3 features

# # Split the tensor into x and y features
# x_features = filtered_tensor[:, :, x_indices]
# y_features = filtered_tensor[:, :, y_indices]

# # Flatten the x and y features for scaling
# x_features_flat = x_features.reshape(-1, x_features.shape[2])
# y_features_flat = y_features.reshape(-1, y_features.shape[2])
# # Standardize the features
# scaler_x = StandardScaler()
# scaler_y = StandardScaler()

# x_features_scaled = scaler_x.fit_transform(x_features_flat)
# y_features_scaled = scaler_y.fit_transform(y_features_flat)

# # Reshape back to the original tensor shape
# x_features_scaled = x_features_scaled.reshape(x_features.shape)
# y_features_scaled = y_features_scaled.reshape(y_features.shape)

# # Split into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x_features_scaled, y_features_scaled, test_size=0.2, random_state=42)

# # Reshape back to original tensor shape
# x_train = x_train.reshape(-1, x_features.shape[1], x_features.shape[2])
# x_test = x_test.reshape(-1, x_features.shape[1], x_features.shape[2])
# y_train = y_train.reshape(-1, y_features.shape[1], y_features.shape[2])
# y_test = y_test.reshape(-1, y_features.shape[1], y_features.shape[2])

# # Example outputs
# print("x_train shape:", x_train.shape)
# print("x_test shape:", x_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)

# # Define a custom learning rate
# learning_rate = 0.001
# optimizer = Adam(learning_rate=learning_rate)

# # Neural network model with Batch Normalization
# model = Sequential()
# model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))  # Define the input shape

# # Add LSTM layer with Batch Normalization
# model.add(LSTM(64, return_sequences=True))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# # Add another LSTM layer with Batch Normalization
# model.add(LSTM(64))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# # Dense layers with Batch Normalization
# model.add(Dense(32))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(y_train.shape[1] * y_train.shape[2]))  # Output layer

# # Compile the model
# model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

# # Train the model
# history = model.fit(x_train, y_train.reshape(-1, y_train.shape[1] * y_train.shape[2]), epochs=50, batch_size=32, validation_split=0.2)

# # Evaluate the model
# mse = model.evaluate(x_test, y_test.reshape(-1, y_test.shape[1] * y_test.shape[2]))
# print(f"Mean Squared Error: {mse}")

# # Make predictions
# predictions = model.predict(x_test)
# predictions = predictions.reshape(-1, y_test.shape[1], y_test.shape[2])

# # Example prediction output
# print("Predictions shape:", predictions.shape)
# print("First prediction:", predictions[0])
# print(y_test[0])

###I_HEAD PREDICTION ####   TRIAL
# new_tensor = [(i[:9] for i in sample[:-1]) for sample in nested_list]#I_head left out for now
new_tensor = [[elem for sublist in sample[:-1] for elem in sublist[:9]] for sample in nested_list]
for i in range(len(new_tensor)):
    new_tensor[i].append(nested_list[i][-1][0])
new_tensor = np.array(new_tensor)

new_x_features=new_tensor[:,:27]
new_y_features=new_tensor[:,-1]

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(new_x_features, new_y_features, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define the neural network model
model = Sequential()
model.add(Input(shape=(27,)))
#     Dense(128, activation='relu'),
model.add(Dense(64,  activation='relu'))
model.add(BatchNormalization())  # Add batch normalization layer
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())  # Add batch normalization layer
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())  # Add batch normalization layer
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))  # Change activation based on your task

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val),batch_size=16, callbacks=[early_stopping])

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')


# previous attempts random forest performs good
# # # Split into training and testing sets
# # x_train, x_test, y_train, y_test = train_test_split(new_x_features, new_y_features, test_size=0.2, random_state=42)
# # print(new_tensor[:10])

# # from sklearn.ensemble import RandomForestRegressor

# # # Initialize and train the model
# # rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# # rf_model.fit(x_train, y_train)

# # # Make predictions
# # y_pred = rf_model.predict(x_test)

# # # Evaluate performance
# # mae = mean_absolute_error(y_test, y_pred)
# # r2 = r2_score(y_test, y_pred)

# # print("Random Forest MAE:", mae)
# # print("Random Forest R²:", r2)

# another appraoch
# # scaler = StandardScaler()
# # x_train = scaler.fit_transform(x_train)
# # x_test = scaler.transform(x_test)

# # model = Sequential([
# #     Input(shape=(27,)),
# #     Dense(128, activation='relu'),
# #     BatchNormalization(),  # Add batch normalization layer
# #     Dropout(0.2),  # Dropout layer for regularization
# #     Dense(128, activation='relu'),
# #     BatchNormalization(),  # Add batch normalization layer
# #     Dropout(0.2),  # Dropout layer for regularization
# #     Dense(64, activation='relu'),
# #     BatchNormalization(),  # Add batch normalization layer
# #     Dense(1)
# # ])


# # # Compile the model
# # model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

# # # Early stopping and learning rate reduction callbacks
# # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# # # Train the model
# # history = model.fit(x_train, y_train, epochs=200, validation_split=0.2, batch_size=64, callbacks=[early_stopping, reduce_lr], verbose=1)

# # # # Train the model
# # # history = model.fit(x_train, y_train, epochs=200, validation_split=0.2)
# # # Evaluate the model on the test set
# # loss, mae = model.evaluate(x_test, y_test)
# # print("Mean Absolute Error (MAE) on test set:", mae)

# # # Make predictions
# # y_pred = model.predict(x_test)

# # # Optional: Calculate additional metrics like R-squared
# # from sklearn.metrics import r2_score

# # r2 = r2_score(y_test, y_pred)
# # print("R-squared (R²) on test set:", r2)

 
