import pandas as pd

column_names = ['0/timestamp', 't', 'no', 'measurement x', 'measurement y', 'reference x', 'reference y']
column_names_test_data = ['0/timestamp', 't', 'no', 'measurement x', 'measurement y', 'reference x', 'reference y',
                          'błąd pomiaru', 'błąd', 'liczba błędnych próbek', '% błędnych próbek']
test_data = pd.read_excel('../resources/pozyxAPI_only_localization_dane_testowe_i_dystrybuanta.xlsx')
test_mes_x = test_data.pop('measurement x')
test_mes_y = test_data.pop('measurement y')
test_tar_x = test_data.pop('reference x')
test_tar_y = test_data.pop('reference y')
test_mes = pd.concat([test_mes_x,test_mes_y], axis=1)
test_tar = pd.concat([test_tar_x,test_tar_y], axis=1)
df1 = pd.read_excel('../resources/pozyxAPI_only_localization_measurement1.xlsx', names=column_names)
df2 = pd.read_excel('../resources/pozyxAPI_only_localization_measurement2.xlsx', names=column_names)
df3 = pd.read_excel('../resources/pozyxAPI_only_localization_measurement3.xlsx', names=column_names)
df4 = pd.read_excel('../resources/pozyxAPI_only_localization_measurement4.xlsx', names=column_names)
df5 = pd.read_excel('../resources/pozyxAPI_only_localization_measurement5.xlsx', names=column_names)
df6 = pd.read_excel('../resources/pozyxAPI_only_localization_measurement6.xlsx', names=column_names)
df7 = pd.read_excel('../resources/pozyxAPI_only_localization_measurement7.xlsx', names=column_names)
df8 = pd.read_excel('../resources/pozyxAPI_only_localization_measurement8.xlsx', names=column_names)
df9 = pd.read_excel('../resources/pozyxAPI_only_localization_measurement9.xlsx', names=column_names)
df10 = pd.read_excel('../resources/pozyxAPI_only_localization_measurement10.xlsx', names=column_names)
df11 = pd.read_excel('../resources/pozyxAPI_only_localization_measurement11.xlsx', names=column_names)
df12 = pd.read_excel('../resources/pozyxAPI_only_localization_measurement12.xlsx', names=column_names)
frames = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12]
df = pd.concat(frames, axis=0, ignore_index=True)
mes_x = df.pop('measurement x')
mes_y = df.pop('measurement y')
training_data = pd.concat([mes_x, mes_y], axis=1)
tar_x = df.pop('reference x')
tar_y = df.pop('reference y')
target_data = pd.concat([tar_x, tar_y], axis=1)
training_data = (training_data.astype('float32') + 2000) / 10000
target_data = (target_data.astype('float32') + 2000) / 10000
test_mes = (test_mes.astype('float32') + 2000) / 10000
test_tar = (test_tar.astype('float32') + 2000) / 10000
