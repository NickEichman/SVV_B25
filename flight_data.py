import numpy as np
import scipy.io as scio

# lh stands for left engine, engine 1
# rh stands for right engine, engine 2
vane_aoa              = np.array(scio.matlab)  # Vane angle of attack, degrees
elevator_dte          = np.array(scio.matlab)  # Deflection of elevator trim, degrees
column_fe             = np.array(scio.matlab)  # Force on elevator control wheel, N
lh_engine_fmf         = np.array(scio.matlab)  # Fuel mass flow, lbs/hr
rh_engine_fmf         = np.array(scio.matlab)  # Fuel mass flow, lbs/hr
lh_engine_itt         = np.array(scio.matlab)  # Inter turbine temperature, deg C
rh_engine_itt         = np.array(scio.matlab)  # Inter turbine temperature, deg C
lh_engine_op          = np.array(scio.matlab)  # Oil pressure, psi
rh_engine_op          = np.array(scio.matlab)  # Oil pressure, psi
lh_engine_fan_n1      = np.array(scio.matlab)  # Fan speed N1, %
rh_engine_fan_n1      = np.array(scio.matlab)  # Fan speed N1, %
lh_engine_turbine_n2  = np.array(scio.matlab)  # Turbine speed N2, %
rh_engine_turbine_n2  = np.array(scio.matlab)  # Turbine speed N2, %
lh_engine_fu          = np.array(scio.matlab)  # Calculated fuel used by fuel mass flow, lbs
rh_engine_fu          = np.array(scio.matlab)  # Calculated fuel used by fuel mass flow, lbs
delta_a               = np.array(scio.matlab)  # Deflection of aileron, deg
delta_e               = np.array(scio.matlab)  # Deflection of elevator, deg
delta_r               = np.array(scio.matlab)  # Deflection of rudder, deg
gps_date              = np.array(scio.matlab)  # UTC date, DD:MM:YY
gps_ut_sec            = np.array(scio.matlab)  # UTC seconds, sec
ahrs1_roll            = np.array(scio.matlab)  # Roll angle, deg
ahrs1_pitch           = np.array(scio.matlab)  # Pitch angle, deg
# fms1_true_heading     = np.array(scio.matlab)  # <no description>, <no units>
gps_lat               = np.array(scio.matlab)  # GNSS Latitude, deg
gps_long              = np.array(scio.matlab)  # GNSS Longitude, deg
ahrs1_b_roll_rate     = np.array(scio.matlab)  # Body roll rate, deg/s
ahrs1_b_pitch_rate    = np.array(scio.matlab)  # Body pitch rate, deg/s
ahrs1_b_yaw_rate      = np.array(scio.matlab)  # Body yaw rate, deg/s
ahrs1_b_long_acc      = np.array(scio.matlab)  # Body longitudinal acceleration, g
ahrs1_b_lat_acc       = np.array(scio.matlab)  # Body lateral acceleration, g
ahrs1_b_norm_acc      = np.array(scio.matlab)  # Body normal acceleration, g
ahrs1_a_hdg_acc       = np.array(scio.matlab)  # Along heading acceleration, g
ahrs1_x_hdg_acc       = np.array(scio.matlab)  # Cross heading acceleration, g
ahrs1_vert_acc        = np.array(scio.matlab)  # Vertical acceleration, g
dadc1_sat             = np.array(scio.matlab)  # Static air temperature, deg C
dadc1_tat             = np.array(scio.matlab)  # Total air temperature, deg C
dadc1_alt             = np.array(scio.matlab)  # Pressure altitude (1013.25 mB), ft
dadc1_bc_alt          = np.array(scio.matlab)  # Baro corrected altitude #1, ft
# dadc1_bc_alt_mb       = np.array(scio.matlab)  # <no description>, <no units>
dadc1_mach            = np.array(scio.matlab)  # Mach number, mach
dadc1_cas             = np.array(scio.matlab)  # Computed airspeed, knots
dadc1_tas             = np.array(scio.matlab)  # True airspeed, knots
dadc1_alt_rate        = np.array(scio.matlab)  # Altitude rate, ft/min
# measurement_running   = np.array(scio.matlab)  # Measurement running, -
# measurement_n_rdy     = np.array(scio.matlab)  # Number of measurements ready, -
# display_graph_state   = np.array(scio.matlab)  # Status of graph, -
# display_active_screen = np.array(scio.matlab)  # Active screen, -
# time                  = np.array(scio.matlab)  # Time, sec
