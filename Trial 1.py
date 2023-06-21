import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Setup_Params
from scipy.optimize import curve_fit


"""

Data preprocessing 

"""


# Import test data initial characterization

init_data = pd.read_excel(r'data_challenge\Initial_characterization.xlsx')

init_data['DateTime'] = pd.to_datetime(init_data['DateTime'], format="%d:%m:%Y %H:%M:%S:%f")

# Remove microseconds from the DateTime column
init_data['DateTime'] = init_data['DateTime'].dt.floor('s')

# Drop duplicate rows based on the DateTime column
init_data.drop_duplicates(subset='DateTime', keep='first', inplace=True)

init_data = init_data.reset_index(drop=True)

print(init_data.head(5))

"""
Extract one cycle Pause/Discharge from dataset
"""

# Find the indices where the state changes from 'pause' to 'discharge' and vice versa
p_d_ind = init_data.index[(init_data['Index'] == 'Pause') & (init_data['Index'].shift(-1) == 'Discharge')]
d_p_ind = init_data.index[(init_data['Index'] == 'Discharge') & (init_data['Index'].shift(-1) == 'Pause')]

# Print the change indices
print("Pause to Discharge Indices:", p_d_ind)
print("Discharge to Pause Indices:", d_p_ind)

# Extract rows of a relaxation cycle
df_cyc = init_data.iloc[p_d_ind[1]+1:d_p_ind[1]-1].reset_index(drop=True)

print(df_cyc.head(5))

"""
New passed time array
"""

# Calculate the time difference from the first entry
time_passed = df_cyc['DateTime'] - df_cyc['DateTime'].iloc[0]

# Convert the time difference to seconds
dt_data = np.array(time_passed.dt.total_seconds())

# Extract the Battery Voltage and current
V_real = np.array(df_cyc['Voltage'])
I_real = np.array(df_cyc['Current'])

"""

Estimate R_0, R_1, R_2, C_1, C_2 values

"""

def parEst(I, k0, k1, k2, a, b):

    n = len(I)
    x = np.zeros(n)
    x[0] = 4.072

    for t in range(1, n):
        x[t] = k0 - k1 * np.exp(-a * t) - k2 * np.exp(-b * t)

    return x



# Use curve_fit to fit the function to the dataset
p0 = [3.8, 0.2, 0.05, 0.01, 0.01]  # Initial parameter guesses
fit_params, _ = curve_fit(parEst, xdata=dt_data, ydata=V_real, p0=p0)

print(fit_params)

# Extract the fitted parameters
k0_fit, k1_fit, k2_fit, a_fit, b_fit = fit_params

# Generate Voltage values using the fitted function
V_fit = parEst(dt_data, k0_fit, k1_fit, k2_fit, a_fit, b_fit)

# Print the fitted parameters
print("Fitted Parameters:")
print(f"k0: {k0_fit}")
print(f"k1: {k1_fit}")
print(f"k2: {k2_fit}")
print(f"a: {a_fit}")
print(f"b: {b_fit}")

# Print the original and fitted x values
# print("Original x values:", V_real)
# print("Fitted x values:", V_fit)

y_mod = parEst(dt_data, k0_fit, k1_fit, k2_fit, a_fit, b_fit)

plt.plot(dt_data, y_mod)
plt.plot(dt_data, V_real)
plt.show()

