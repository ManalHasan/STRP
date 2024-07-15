import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Input, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Filter out the single-element lists
filtered_tensor = [sample[:-1] for sample in nested_list]
filtered_tensor = np.array(filtered_tensor)
# Check the structure after filtering
print(f"Filtered tensor shape: {filtered_tensor.shape}")

# Define indices for splitting into x and y features
x_indices = slice(0, 9)  # First 9 features
y_indices = slice(9, 12)  # Next 3 features

# Split the tensor into x and y features
x_features = filtered_tensor[:, :, x_indices]
y_features = filtered_tensor[:, :, y_indices]

# Flatten the x and y features for scaling
x_features_flat = x_features.reshape(-1, x_features.shape[2])
y_features_flat = y_features.reshape(-1, y_features.shape[2])

# Standardize the features
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_features_scaled = scaler_x.fit_transform(x_features_flat)
y_features_scaled = scaler_y.fit_transform(y_features_flat)

# Reshape back to the original tensor shape
x_features_scaled = x_features_scaled.reshape(x_features.shape)
y_features_scaled = y_features_scaled.reshape(y_features.shape)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_features_scaled, y_features_scaled, test_size=0.2, random_state=42)

# Reshape back to original tensor shape
x_train = x_train.reshape(-1, x_features.shape[1], x_features.shape[2])
x_test = x_test.reshape(-1, x_features.shape[1], x_features.shape[2])
y_train = y_train.reshape(-1, y_features.shape[1], y_features.shape[2])
y_test = y_test.reshape(-1, y_features.shape[1], y_features.shape[2])

# Example outputs
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Define a custom learning rate
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

# Neural network model with Batch Normalization
model = Sequential()
model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))  # Define the input shape

# Add LSTM layer with Batch Normalization
model.add(LSTM(64, return_sequences=True))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Add another LSTM layer with Batch Normalization
model.add(LSTM(64))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Dense layers with Batch Normalization
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(y_train.shape[1] * y_train.shape[2]))  # Output layer

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

# Train the model
history = model.fit(x_train, y_train.reshape(-1, y_train.shape[1] * y_train.shape[2]), epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
mse = model.evaluate(x_test, y_test.reshape(-1, y_test.shape[1] * y_test.shape[2]))
print(f"Mean Squared Error: {mse}")

# Make predictions
predictions = model.predict(x_test)
predictions = predictions.reshape(-1, y_test.shape[1], y_test.shape[2])

# Example prediction output
print("Predictions shape:", predictions.shape)
print("First prediction:", predictions[0])
print(y_test[0])


# # import pandas as pd
# # import numpy as np

# # # Define the dataset
# # data = [
# #     [
# #         [0.2, 0.3, 0.5, 0.0, 0.7, 0.3, 60.0, 70.0],
# #         [0.2, 0.3, 0.5, 0.0, 0.7, 0.3, 90.0, 40.0],
# #         [0.2, 0.3, 0.5, 0.0, 0.7, 0.3, 120.0, 80.0],
# #         [0.2, 0.3, 0.5, 0.0, 0.7, 0.3, 60.0, 30.0]
# #     ],
# #     [
# #         [0.2, 0.3, 0.5, 0.0, 0.7, 0.3, 60.0, 70.0],
# #         [0.2, 0.3, 0.5, 0.0, 0.7, 0.3, 90.0, 40.0],
# #         [0.2, 0.3, 0.5, 0.0, 0.7, 0.3, 120.0, 80.0],
# #         [0.2, 0.3, 0.5, 0.0, 0.7, 0.3, 60.0, 30.0]
# #     ]
# # ]

# # # Flatten the nested list structure and separate features and targets
# # flat_data = []
# # for sublist in data:
# #     for sample in sublist:
# #         flat_data.append(sample)

# # # Split features and targets
# # features = [sample[:-2] for sample in flat_data]
# # targets = [sample[-2:] for sample in flat_data]

# # # Create a DataFrame
# # columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'target1', 'target2']
# # df = pd.DataFrame(flat_data, columns=columns)

# # # print("DataFrame:")
# # # print(df)
# # # Save the DataFrame to a CSV file
# # csv_file = 'multidimensional_dataset.csv'
# # df.to_csv(csv_file, index=False)
# # print(f"DataFrame has been saved to {csv_file}")
# # import tensorflow as tf

# # # Read the DataFrame
# # df = pd.read_csv('multidimensional_dataset.csv')

# # # Separate features and targets
# # x = df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']].values
# # y = df[['target1', 'target2']].values

# # # Convert to TensorFlow datasets
# # x_dataset = tf.data.Dataset.from_tensor_slices(x)
# # y_dataset = tf.data.Dataset.from_tensor_slices(y)

# # # Combine features and labels into one dataset
# # dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

# # # Verify the dataset
# # for features, labels in dataset:
# #     print("Features:", features.numpy(), "Labels:", labels.numpy())


 