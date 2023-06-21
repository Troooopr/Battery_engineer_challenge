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



def est_par(i, R0, R1, R2, C1, C2, start_SoC):

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


    # Equations 2nd order ECM

    A = np.matrix([[np.exp(-dt / (R1 * C1)), 0], [0, np.exp(-dt / (R2 * C2))]])
    # B = np.matrix([dt / C1, dt / C2])
    B = np.matrix([R1 * (1 - np.exp(-dt / (R1 * C1))), R2 * (1 - np.exp(-dt / (R2 * C2)))])
    C = np.array([1, 1])

    x = np.zeros((2, n))
    x = np.matrix([U_RC1, U_RC2])
    u = i

    for t in range(1, n):
        # x[:, t] = A * x[:, t - 1] + B.T * u[t]

        x[:, t] = A * x[:, t - 1] + B.T * u[t]

        # For discharge: Ampere counting method using OCV
        if u[t] <= 0:  # discharging
            C_Ah[t] = C_Ah[t - 1] + i[t] * (dt / 3600)
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
            # print(C * x[:, t])

            U_bat[t] = C * x[:, t] + U_OCV[t]

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

# Print the change indices
print("Pause to Discharge Indices:", p_d_ind)
print("Discharge to Pause Indices:", d_p_ind)


def get_start_parameter():
    '''
    Definition to get start parameters
    '''

    r0_min = 0.0000005
    R0 = 0.005
    r0_max = 0.01
    r1_min = 0.00001
    R1 = 0.005
    r1_max = 0.01
    r2_min = 0.00001
    R2 = 0.005
    r2_max = 0.01
    c1_min = 10
    C1 = 200
    c1_max = 100000
    c2_min = 1000
    C2 = 2000
    c2_max = 10000000

    p0 = [R0, R1, R2, C1, C2]
    bounds_min = [r0_min, r1_min, r2_min, c1_min, c2_min]
    bounds_max = [r0_max, r1_max, r2_max, c1_max, c2_max]

    return p0, bounds_min, bounds_max



for m in range(0, len(p_d_ind) - 1):

    # Extract rows of a relaxation cycle
    cyc_data = init_data.iloc[p_d_ind[m]:p_d_ind[m + 1] - 1].reset_index(drop=True)

    # Extract Voltage, Current and Start SoC of Cycle
    i_real = np.array(cyc_data['Current'])
    u_real = np.array(cyc_data['Voltage'])
    start_SOC = cyc_data.iloc[0]['State_of_Charge'] / 100

    # Use curve_fit to fit the ECM to the dataset
    p0, bounds_min, bounds_max = get_start_parameter()
    fit_params, _ = curve_fit(lambda i, R0, R1, R2, C1, C2: est_par(i, R0, R1, R2, C1, C2, start_SOC), i_real, u_real,
                              p0=p0, bounds=(bounds_min, bounds_max))

    # Extract the fitted parameters
    R0_fit, R1_fit, R2_fit, C1_fit, C2_fit = fit_params

    # Generate Voltage values using the fitted function
    V_fit = est_par(i_real, R0_fit, R1_fit, R2_fit, C1_fit, C2_fit, start_SOC)

    # Print the fitted parameters
    print("Fitted Parameters:")
    print(f"R0: {R0_fit}")
    print(f"R1: {R1_fit}")
    print(f"R2: {R2_fit}")
    print(f"C1: {C1_fit}")
    print(f"C2: {C2_fit}")

    # Print the original and fitted x values
    print("Original x values:", u_real)
    print("Fitted x values:", V_fit)

    plt.plot(cyc_data['DateTime'], V_fit)
    plt.plot(cyc_data['DateTime'], u_real)
    plt.legend(['Vfit', 'ureal'])

plt.show()



