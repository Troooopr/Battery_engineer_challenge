import numpy as np
import matplotlib.pyplot as plt
import Setup_Params
import pandas as pd
from scipy.optimize import curve_fit

aging_df = pd.read_excel(r'data_challenge/cycles.xlsx')

aging_df['DateTime'] = pd.to_datetime(aging_df['DateTime'], format="%d:%m:%Y %H:%M")
aging_df = aging_df.sort_values('DateTime')

# plt.plot(aging_df['DateTime'], aging_df['Voltage'])
# plt.show()

# plt.plot(aging_df['DateTime'], aging_df['Current'])
# plt.show()

# plt.plot(aging_df['DateTime'], aging_df['Capacity'])
# plt.show()

# Drop rows where 'Current' = 0
aging_df = aging_df.drop(aging_df[aging_df['Current'] == 0].index).reset_index(drop=True)

# Calculate time differences between consecutive rows
time_diff = aging_df['DateTime'].diff()

# Find the indices where a gap in DateTime occurs
gap_indices = time_diff[time_diff > pd.Timedelta(hours=1)].index
print(gap_indices)

# Split the DataFrame based on the identified gap indices
cycles = []

if len(gap_indices) > 0:
    start_index = 0

    for index in gap_indices:
        cycle_df = aging_df[start_index:index].reset_index(drop=True)
        cycles.append(cycle_df)
        start_index = index + 1

    cycle_df = aging_df[start_index:].reset_index(drop=True)
    cycles.append(cycle_df)
else:
    cycle_df = aging_df.reset_index(drop=True)
    cycles.append(cycle_df)

# Remove Outlier
del cycles[0]

SoH = np.zeros(len(cycles))

max_Capacity = np.zeros(len(cycles))
max_Voltage = np.zeros(len(cycles))
min_Voltage = np.zeros(len(cycles))
start_Voltage = np.zeros(len(cycles))

# Plot each part with respect to the 'Current' column
for i, cycle in enumerate(cycles):
    cycle = cycle.drop(cycle.index[-1])
    # plt.plot(cycle['Capacity'], cycle['Voltage'], label=f'Cycle {i + 1}')


    # Calculate delta Capacity
    max_Capacity[i] = cycle['Capacity'].max()
    # print(f"Max Capacity for Cycle {i + 1}: {max_Capacity}\n")

    # Calculate max/min Voltage
    max_Voltage[i] = cycle['Voltage'].max()
    min_Voltage[i] = cycle['Voltage'].min()
    start_Voltage[i] = cycle['Voltage'].loc[0]



    SoC = np.zeros((len(cycle)))
    SoC[0] = Setup_Params.fct_int_Soc(start_Voltage[i])
    print(SoC[0])

    # Calculate SoC
    for k in range(1, len(cycle)):
        # R_col = 0.000634
        C_bat = 56.8
        i_k_1 = cycle['Current'].loc[k - 1]
        # u_k_1 = cycle['Voltage'].loc[k - 1]
        Delta_SOC = i_k_1 / (600 * C_bat)
        SoC[k] = SoC[k - 1] + Delta_SOC
        # SoC[k] = SoC[k-1] + (cycle['Capacity'].loc[k] - cycle['Capacity'].loc[k - 1]) / C_bat

    plt.plot(SoC, cycle['Voltage'], label=f'Cycle {i + 1}')


    # Calculate SoH
    # SoH[i] = max_Capacity[i]
    # SoH[i] = max_Capacity[i] / cycles[0]['Capacity'].max()
    # SoH[i] =

    print(SoC)

print(max_Voltage)
print(min_Voltage)
# print(max_Capacity)

print(f"SoH for Cycle: {SoH}\n")

# Add labels and legend to the plot
plt.xlabel('SOC [-]')
plt.ylabel('Voltage [V]')
plt.title('Aging impact on Voltage/SOC behavior')
plt.legend()

# Display the plot
plt.show()


# plt.plot(SoH)
# plt.show()
