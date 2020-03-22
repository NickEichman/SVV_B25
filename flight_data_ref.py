import numpy as np
import scipy.io as scio

## Imports and slices input from the reference flight data, in all of the eigenmotion flight sections
# flight sections

phugoid_time_start = 32370
phugoid_time_end = 35500

aperiodic_roll_time_start = 35500
aperiodic_roll_time_end = 36350

short_period_time_start = 36350
short_period_time_end = 37170

dutch_roll_time_start = 37170
dutch_roll_time_end = 37670

dutch_roll_YD_time_start = 37670
dutch_roll_YD_time_end = 39200

spiral_time_start = 39200
spiral_time_end = 41000  # Guesstimate

# lh stands for left engine, engine 1
# rh stands for right engine, engine 2
input_file            = scio.loadmat('matlab.mat')
vane_aoa              = np.array(input_file['flightdata'][0][0][0][0][0][0])  # Vane angle of attack, degrees
elevator_dte          = np.array(input_file['flightdata'][0][0][1][0][0][0])  # Deflection of elevator trim, degrees
column_fe             = np.array(input_file['flightdata'][0][0][2][0][0][0])  # Force on elevator control wheel, N
lh_engine_fmf         = np.array(input_file['flightdata'][0][0][3][0][0][0])  # Fuel mass flow, lbs/hr
rh_engine_fmf         = np.array(input_file['flightdata'][0][0][4][0][0][0])  # Fuel mass flow, lbs/hr
lh_engine_itt         = np.array(input_file['flightdata'][0][0][5][0][0][0])  # Inter turbine temperature, deg C
rh_engine_itt         = np.array(input_file['flightdata'][0][0][6][0][0][0])  # Inter turbine temperature, deg C
lh_engine_op          = np.array(input_file['flightdata'][0][0][7][0][0][0])  # Oil pressure, psi
rh_engine_op          = np.array(input_file['flightdata'][0][0][8][0][0][0])  # Oil pressure, psi
lh_engine_fan_n1      = np.array(input_file['flightdata'][0][0][9][0][0][0])  # Fan speed N1, %
rh_engine_fan_n1      = np.array(input_file['flightdata'][0][0][11][0][0][0])  # Fan speed N1, %
lh_engine_turbine_n2  = np.array(input_file['flightdata'][0][0][10][0][0][0])  # Turbine speed N2, %
rh_engine_turbine_n2  = np.array(input_file['flightdata'][0][0][12][0][0][0])  # Turbine speed N2, %
lh_engine_fu          = np.array(input_file['flightdata'][0][0][13][0][0][0])  # Calculated fuel used, lbs
rh_engine_fu          = np.array(input_file['flightdata'][0][0][14][0][0][0])  # Calculated fuel used, lbs
delta_a               = np.array(input_file['flightdata'][0][0][15][0][0][0])  # Deflection of aileron, deg
delta_e               = np.array(input_file['flightdata'][0][0][16][0][0][0])  # Deflection of elevator, deg
delta_r               = np.array(input_file['flightdata'][0][0][17][0][0][0])  # Deflection of rudder, deg
gps_date              = np.array(input_file['flightdata'][0][0][18][0][0][0])  # UTC date, DD:MM:YY
gps_ut_sec            = np.array(input_file['flightdata'][0][0][19][0][0][0])  # UTC seconds, sec
ahrs1_roll            = np.array(input_file['flightdata'][0][0][20][0][0][0])  # Roll angle, deg
ahrs1_pitch           = np.array(input_file['flightdata'][0][0][21][0][0][0])  # Pitch angle, deg
fms1_true_heading     = np.array(input_file['flightdata'][0][0][22][0][0][0])  # <no description>, <no units>
gps_lat               = np.array(input_file['flightdata'][0][0][23][0][0][0])  # GNSS Latitude, deg
gps_long              = np.array(input_file['flightdata'][0][0][24][0][0][0])  # GNSS Longitude, deg
ahrs1_b_roll_rate     = np.array(input_file['flightdata'][0][0][25][0][0][0])  # Body roll rate, deg/s
ahrs1_b_pitch_rate    = np.array(input_file['flightdata'][0][0][26][0][0][0])  # Body pitch rate, deg/s
ahrs1_b_yaw_rate      = np.array(input_file['flightdata'][0][0][27][0][0][0])  # Body yaw rate, deg/s
ahrs1_b_long_acc      = np.array(input_file['flightdata'][0][0][28][0][0][0])  # Body longitudinal acceleration, g
ahrs1_b_lat_acc       = np.array(input_file['flightdata'][0][0][29][0][0][0])  # Body lateral acceleration, g
ahrs1_b_norm_acc      = np.array(input_file['flightdata'][0][0][30][0][0][0])  # Body normal acceleration, g
ahrs1_a_hdg_acc       = np.array(input_file['flightdata'][0][0][31][0][0][0])  # Along heading acceleration, g
ahrs1_x_hdg_acc       = np.array(input_file['flightdata'][0][0][32][0][0][0])  # Cross heading acceleration, g
ahrs1_vert_acc        = np.array(input_file['flightdata'][0][0][33][0][0][0])  # Vertical acceleration, g
dadc1_sat             = np.array(input_file['flightdata'][0][0][34][0][0][0])  # Static air temperature, deg C
dadc1_tat             = np.array(input_file['flightdata'][0][0][35][0][0][0])  # Total air temperature, deg C
dadc1_alt             = np.array(input_file['flightdata'][0][0][36][0][0][0])  # Pressure altitude (1013.25 mB), ft
dadc1_bc_alt          = np.array(input_file['flightdata'][0][0][37][0][0][0])  # Baro corrected altitude #1, ft
dadc1_bc_alt_mb       = np.array(input_file['flightdata'][0][0][38][0][0][0])  # <no description>, <no units>
dadc1_mach            = np.array(input_file['flightdata'][0][0][39][0][0][0])  # Mach number, mach
dadc1_cas             = np.array(input_file['flightdata'][0][0][40][0][0][0])  # Computed airspeed, knots
dadc1_tas             = np.array(input_file['flightdata'][0][0][41][0][0][0])  # True airspeed, knots
dadc1_alt_rate        = np.array(input_file['flightdata'][0][0][42][0][0][0])  # Altitude rate, ft/min
measurement_running   = np.array(input_file['flightdata'][0][0][43][0][0][0])  # Measurement running, -
measurement_n_rdy     = np.array(input_file['flightdata'][0][0][44][0][0][0])  # Number of measurements ready, -
display_graph_state   = np.array(input_file['flightdata'][0][0][45][0][0][0])  # Status of graph, -
display_active_screen = np.array(input_file['flightdata'][0][0][46][0][0][0])  # Active screen, -
time                  = np.array(input_file['flightdata'][0][0][47][0][0][0])  # Time, sec
length_array          = np.arange(0, len(vane_aoa))                                # For plotting purposes
totalfuelflow = lh_engine_fmf+rh_engine_fmf


# Function to get all pertaining data for the eigenmotions. n is the eigenmotion chosen, 0 is short period, 1 phugoid,
# 2 dutch roll, 3 dutch roll yaw dampening, 4 aperiodic roll and 5 spiral. Returns the following numpy arrays, in order:
# Angle of attack, true air speed, roll angle, pitch angle, yaw angle? it is not certain what it is from the matlab file
# roll rate, pitch rate, yaw rate, elevator trim deflection, aileron deflection, elevator deflection, rudder deflection
def get_data_eigen(n):
    if n == 0:
        start = short_period_time_start
        end = short_period_time_end
    elif n == 1:
        start = phugoid_time_start
        end = phugoid_time_end
    elif n == 2:
        start = dutch_roll_time_start
        end = dutch_roll_time_end
    elif n == 3:
        start = dutch_roll_YD_time_start
        end = dutch_roll_YD_time_end
    elif n == 4:
        start = aperiodic_roll_time_start
        end = aperiodic_roll_time_end
    elif n == 5:
        start = spiral_time_start
        end = spiral_time_end
    aoa = vane_aoa[start:end]
    tas = dadc1_tas[start:end]
    roll = ahrs1_roll[start:end]
    pitch = ahrs1_pitch[start:end]
    yaw = fms1_true_heading[start:end]
    roll_rate = ahrs1_b_roll_rate[start:end]
    pitch_rate = ahrs1_b_pitch_rate[start:end]
    yaw_rate = ahrs1_b_yaw_rate[start:end]
    aileron_def = delta_a[start:end]
    elevator_def = delta_e[start:end]
    rudder_def = delta_r[start:end]
    pressure_alt = dadc1_alt[start:end]
    time_slice = np.arange(0, (end - start) * 0.1, 0.1)

    return [aoa, tas, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, aileron_def, elevator_def, rudder_def,
            pressure_alt], time_slice
