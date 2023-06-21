import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import test data initial characterization

init_data = pd.read_excel(r'data_challenge/Initial_characterization.xlsx')

init_data['DateTime'] = pd.to_datetime(init_data['DateTime'], format="%d:%m:%Y %H:%M:%S:%f")

R_C_data = pd.read_excel(r'data_challenge\Parameter_ECM.xlsx')

# Remove microseconds from the DateTime column
init_data['DateTime'] = init_data['DateTime'].dt.floor('s')

"""

Calculate Battery Capacity

"""

C_bat = init_data['Charge_Throughput'].iloc[-1] - init_data['Charge_Throughput'].iloc[0]
# print('C_Ah = ', C_bat)
E_bat = init_data['Energy_Throughput'].iloc[-1] - init_data['Energy_Throughput'].iloc[0]
# print('E_Wh = ', E_bat)

"""

Estimate OCV values

"""

# get indices for each row that changes Index from Pause
pause_end_indices = init_data[init_data['Index'].shift(-1) != 'Pause'].loc[init_data['Index'] == 'Pause'].index

# Extract the rows with the pause_end_indices
OCV_df = init_data[['State_of_Charge', 'Voltage']].loc[pause_end_indices]

# Linear interpolation function of the SoC/OCV dependency
OCV = np.array(OCV_df)
OCV[:, 0] = OCV[:, 0] / 100

fct_int_OCV = interp1d(OCV[:, 0], OCV[:, 1], fill_value='extrapolate')
fct_int_Soc = interp1d(OCV[:, 1], OCV[:, 0], fill_value='extrapolate')

"""
# Plot OCV w.r.t SOC
plt.plot(OCV[:, 0], OCV[:, 1])
plt.xlabel('SOC [-]', fontsize=24)
plt.ylabel('OCV [V]', fontsize=24)
plt.title('OCV behaviour discharge @ 50A', fontsize=28)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
"""
# Display the plot
plt.show()


"""

Estimate R/C values and linear interpolate them w.r.t. SoC

"""

# Extract values from Excel
SOC = np.array(R_C_data['SoC'])/100
R0 = np.array(R_C_data['R0'])
R1 = np.array(R_C_data['R1'])
R2 = np.array(R_C_data['R2'])
C1 = np.array(R_C_data['C1'])
C2 = np.array(R_C_data['C2'])

# interpolate Constants
fct_int_R0 = interp1d(SOC, R0, fill_value='extrapolate')
fct_int_R1 = interp1d(SOC, R1, fill_value='extrapolate')
fct_int_R2 = interp1d(SOC, R2, fill_value='extrapolate')
fct_int_C1 = interp1d(SOC, C1, fill_value='extrapolate')
fct_int_C2 = interp1d(SOC, C2, fill_value='extrapolate')
"""
# Plot R and Cs
# Plot R0 vs SOC
plt.figure()
plt.plot(SOC, R0)
plt.xlabel('SOC [-]', fontsize=24)
plt.ylabel('R0 [Ohm]', fontsize=24)
plt.title('R0 vs SOC', fontsize=28)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Plot R1 vs SOC
plt.figure()
plt.plot(SOC, R1)
plt.xlabel('SOC [-]', fontsize=24)
plt.ylabel('R1 [Ohm]', fontsize=24)
plt.title('R1 vs SOC', fontsize=28)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Plot R2 vs SOC
plt.figure()
plt.plot(SOC, R2)
plt.xlabel('SOC [-]', fontsize=24)
plt.ylabel('R2 [Ohm]', fontsize=24)
plt.title('R2 vs SOC', fontsize=28)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Plot C1 vs SOC
plt.figure()
plt.plot(SOC, C1)
plt.xlabel('SOC [-]', fontsize=24)
plt.ylabel('C1 [F]', fontsize=24)
plt.title('C1 vs SOC', fontsize=28)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Plot C2 vs SOC
plt.figure()
plt.plot(SOC, C2)
plt.xlabel('SOC [-]', fontsize=24)
plt.ylabel('C2 [F]', fontsize=24)
plt.title('C2 vs SOC', fontsize=28)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
"""

