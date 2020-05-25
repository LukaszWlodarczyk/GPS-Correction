import glob
import os

import pandas as pd
import tensorflow as tf
import numpy as np

# Read test data from file
test_data = pd.read_excel('../resources/pozyxAPI_only_localization_dane_testowe_i_dystrybuanta.xlsx')
test_mes_x = test_data.pop('measurement x')
test_mes_y = test_data.pop('measurement y')
test_tar_x = test_data.pop('reference x')
test_tar_y = test_data.pop('reference y')
test_mes = pd.concat([test_mes_x, test_mes_y], axis=1)
test_tar = pd.concat([test_tar_x, test_tar_y], axis=1)

# Read training & targed data from files
path = '../resources/'
column_names = ['0/timestamp', 't', 'no', 'measurement x', 'measurement y', 'reference x', 'reference y']

all_files = glob.glob(os.path.join(path, 'pozyxAPI_only_localization_measurement*.xlsx'))
df_from_each_file = (pd.read_excel(f, names=column_names) for f in all_files)
concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

mes_x = concatenated_df.pop('measurement x')
mes_y = concatenated_df.pop('measurement y')
training_data = pd.concat([mes_x, mes_y], axis=1)

tar_x = concatenated_df.pop('reference x')
tar_y = concatenated_df.pop('reference y')
target_data = pd.concat([tar_x, tar_y], axis=1)

# Add offset to data for better readability
training_data = (training_data.astype('float32') + 2000) / 10000
target_data = (target_data.astype('float32') + 2000) / 10000

# Data for validation
training_data_1 = training_data[:(11 * 1540)]
target_data_1 = target_data[:(11 * 1540)]
val_data_mes = training_data[1540 * 11:]
val_data_tar = target_data[1540 * 11:]
val_data_mes.reset_index(drop=True, inplace=True)
val_data_tar.reset_index(drop=True, inplace=True)

# Add offset to data for better readability
test_mes = (test_mes.astype('float32') + 2000) / 10000
test_tar = (test_tar.astype('float32') + 2000) / 10000

#Create network
network = tf.keras.models.Sequential()
network.add(tf.keras.layers.Dense(128, activation='relu'))
network.add(tf.keras.layers.Dense(64, activation='relu'))
network.add(tf.keras.layers.Dense(32, activation='relu'))
network.add(tf.keras.layers.Dense(16, activation='relu'))
network.add(tf.keras.layers.Dense(8, activation='relu'))
network.add(tf.keras.layers.Dense(2, activation='sigmoid'))

# Compile network using Adam optimizer
network.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['accuracy'])

# Fit (train) network, calculate it's accuracy using val_data_mes and val_data_tar as validation data set
network.fit(np.asarray(training_data_1), np.asarray(target_data_1), epochs=150, batch_size=512,
            validation_data=(val_data_mes, val_data_tar))

# Evaluate network to get loss value % metrics values
network.evaluate(test_mes, test_tar, batch_size=512)

#Get each layer neuron weights and save to file
weights = network.layers[0].get_weights()[0]
print(weights)

# Get output for desired measurements using trained network
res = network.predict(test_mes)
res = res * 10000 - 2000
res1 = pd.DataFrame(res)
#res1.to_excel('wyniki1.xlsx', engine='xlsxwriter')
print(res * 10000 - 2000)
