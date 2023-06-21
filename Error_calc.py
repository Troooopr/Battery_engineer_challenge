import numpy as np
import matplotlib.pyplot as plt
import Setup_Params
import pandas as pd
from scipy.optimize import curve_fit

# Import test data initial characterization

init_data = pd.read_excel(r'data_challenge\Initial_characterization.xlsx')

init_data['DateTime'] = pd.to_datetime(init_data['DateTime'], format="%d:%m:%Y %H:%M:%S:%f")

# Remove microseconds from the DateTime column
init_data['DateTime'] = init_data['DateTime'].dt.floor('s')

# Drop duplicate rows based on the DateTime column
init_data.drop_duplicates(subset='DateTime', keep='first', inplace=True)

init_data = init_data.reset_index(drop=True)

print(init_data.head(20))



def ecm(i, start_SoC):

    # Setup parameter
    START_SOC = start_SoC
    dt = 1
    C_bat = 56.8

    n = len(i)
    U_RC1 = np.zeros(n, )
    U_RC2 = np.zeros(n, )
    U_OCV = np.zeros(n, )
    U_R0 = np.zeros(n, )
    U_OCV[0] = Setup_Params.fct_int_OCV(start_SoC)
    C_Ah = np.zeros(n, )
    C_Ah[0] = START_SOC * C_bat
    U_bat = np.zeros(n, )
    U_bat[0] = Setup_Params.fct_int_OCV(start_SoC)
    SOC = np.zeros(n, )
    SOC[0] = START_SOC

    #  R_coul = 0.00063


    # Equations 2nd order ECM

    x = np.matrix([U_RC1, U_RC2])
    u = i

    for t in range(1, n):

        R0 = Setup_Params.fct_int_R0(SOC[t - 1])
        R1 = Setup_Params.fct_int_R1(SOC[t - 1])
        R2 = Setup_Params.fct_int_R2(SOC[t - 1])
        C1 = Setup_Params.fct_int_C1(SOC[t - 1])
        C2 = Setup_Params.fct_int_C2(SOC[t - 1])

        # Equations 2nd order ECM
        A = np.matrix([[np.exp(-dt / (R1 * C1)), 0], [0, np.exp(-dt / (R2 * C2))]])
        # B = np.matrix([dt / C1, dt / C2])
        B = np.matrix([R1 * (1 - np.exp(-dt / (R1 * C1))), R2 * (1 - np.exp(-dt / (R2 * C2)))])
        C = np.array([1, 1])


        x[:, t] = A * x[:, t - 1] + B.T * u[t]


        # For discharge: Ampere counting method using OCV
        if u[t] <= 0:  # discharging
            C_Ah[t] = C_Ah[t - 1] + i[t] * (dt / 3600)  # - i[t]**2 * R_coul / 3600
            # print(C_Ah[t])
            if (C_Ah[t] / C_bat) >= 0.0024:
                U_OCV[t] = Setup_Params.fct_int_OCV((C_Ah[t] / C_bat))
            else:
                U_OCV[t] = Setup_Params.fct_int_OCV(0.0024)

            U_bat[t] = C * x[:, t] + R0 * u[t] + U_OCV[t]

        # For Pause:
        if u[t] == 0:  # Pause
            C_Ah[t] = C_Ah[t - 1]
            U_OCV[t] = Setup_Params.fct_int_OCV((C_Ah[t] / C_bat))

            U_bat[t] = C * x[:, t] + U_OCV[t]

        SOC[t] = (C_Ah[t] / C_bat)
        # U_RC1[t] = -x[0, t]
        # U_RC2[t] = -x[1, t]
        # U_R0[t] = R0 * u[t]
        # print('U_RC1', U_RC1[t])

    # print(x)
    return U_bat

"""
Extract one cycle Pause/Discharge from dataset
"""

# Find the indices where the state changes from 'pause' to 'discharge' and vice versa
p_d_ind = init_data.index[(init_data['Index'] == 'Pause') & (init_data['Index'].shift(-1) != 'Pause')]
d_p_ind = init_data.index[(init_data['Index'] == 'Discharge') & (init_data['Index'].shift(-1) != 'Discharge')]

# Extract rows of a relaxation cycle
# cyc_data = init_data.iloc[p_d_ind[0]:p_d_ind[-1] - 1].reset_index(drop=True)
cyc_data = init_data

# Extract Voltage, Current and Start SoC of Cycle
i_real = np.array(cyc_data['Current'])
u_real = np.array(cyc_data['Voltage'])
start_SOC = cyc_data.iloc[0]['State_of_Charge'] / 100

# Generate Voltage values using the fitted function
u_fit = ecm(i_real, start_SOC)

plt.plot(cyc_data['DateTime'], u_fit)
plt.plot(cyc_data['DateTime'], u_real)
plt.legend(['ufit', 'ureal'])
plt.xlabel('Time [s]')
plt.ylabel('Battery Voltage U_bat [V]')
plt.title('Measured and simulated Voltage')

plt.show()

"""
Calculate and plot the error
"""

# Calculate the error (residuals)
error = u_real - u_fit
plt.plot(cyc_data['DateTime'], error)
plt.xlabel('Time [s]')
plt.ylabel('Error [V]')
plt.title('Voltage Error')
plt.show()

# Plot the error distribution
plt.hist(error, bins=20)
plt.xlabel('Error [V]')
plt.ylabel('Frequency [-]')
plt.title('Voltage Error Distribution')
plt.show()


