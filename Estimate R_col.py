import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import test data initial characterization

init_data = pd.read_excel(r'data_challenge\Initial_characterization.xlsx')

init_data['DateTime'] = pd.to_datetime(init_data['DateTime'], format="%d:%m:%Y %H:%M:%S:%f")

# Remove microseconds from the DateTime column
init_data['DateTime'] = init_data['DateTime'].dt.floor('s')

# Drop duplicate rows based on the DateTime column
init_data.drop_duplicates(subset='DateTime', keep='first', inplace=True)

print(init_data.head(30))

"""

Estimate R0 values

"""

# get indices for each row that changes Index to Discharge
discharge_start_indices = init_data[init_data['Index'].shift(1) != 'Discharge'].loc[init_data['Index'] == 'Discharge'].index
print(discharge_start_indices)

# Get the DateTime values of the filtered indices
filtered_datetimes = init_data.loc[discharge_start_indices, 'DateTime'].reset_index(drop=True)
print(filtered_datetimes)

# Get the corresponding voltage values
voltages_0s = init_data[init_data['DateTime'].isin(filtered_datetimes)]['Voltage'].reset_index(drop=True)
print(voltages_0s)

# Add 10 seconds to each DateTime value
target_datetimes = filtered_datetimes + pd.Timedelta(seconds=10)
print(target_datetimes)

# Get the corresponding voltage values
voltages_10s = init_data[init_data['DateTime'].isin(target_datetimes)]['Voltage'].reset_index(drop=True)
print(voltages_10s)

# Create a new DataFrame with the DateTime and Voltage values
result = pd.DataFrame({'StartDateTime': filtered_datetimes, 'EndDateTime': target_datetimes, 'Voltage0s': voltages_0s,
                       'Voltage10s': voltages_10s})

# Calculate R0
result['R0'] = (result['Voltage0s'] - result['Voltage10s'])/50

# Estimate average R0
av_R0 = np.average(result['R0'])

# Print the result
print(av_R0)

plt.plot(result['R0'])
plt.show()

