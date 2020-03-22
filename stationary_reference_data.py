import numpy as np

bem = 9165*0.453592
m_fuel = 4050*0.453592                                              # [kg]
masses = np.array([95,92,74,66,61,75,78,86,68])                     # [kg]
takeoff_mass = np.sum(masses) + m_fuel + bem                        # [kg]
g = 9.81

# Stationary Measurements 1

sm1_time = np.array([1157,1297,1426,1564,1787,1920])                # [s]
sm1_alt = np.array([5010,5020,5020,5030,5020,5110])*0.3048          # [m]
sm1_IAS = np.array([249,221,192,163,130,118])*0.514444              # [m/s]
sm1_alpha = np.array([1.7,2.4,3.6,5.4,8.7,10.6])*np.pi/180          # [rad]
sm1_FFl = np.array([798,673,561,463,443,474])*0.453592/(3600)       # [kg/s]
sm1_FFr = np.array([813,682,579,484,467,499])*0.453592/(3600)       # [kg/s]
sm1_F_used = np.array([360,412,447,478,532,570])*0.453592           # [kg]
sm1_temp = np.array([12.5,10.5,8.8,7.2,6,5.2])+273.15               # [K]
sm1_weight = (takeoff_mass - sm1_F_used) * g



# Stationary Measurements 2

sm2_time = np.array([2136,2252,2328,2402,2518,2724,2799])               # [s] TODO update
sm2_alt = np.array([6060,6350,6550,6880,6160,5810,5310])*0.3048         # [m]
sm2_IAS = np.array([161,150,140,130,173,179,192])*0.514444              # [m/s]
sm2_alpha = np.array([5.3,6.3,7.3,8.5,4.5,4.1,3.4])*np.pi/180           # [rad]
sm2_delta_e = np.array([0,-0.4,-0.9,-1.5,0.4,0.6,1])*np.pi/180          # [rad]
sm2_delta_tr = np.array([2.8,2.8,2.8,2.8,2.8,2.8,2.8])*np.pi/180        # [rad]
sm2_Fe = np.array([0,-23,-29,-46,26,40,83])                             # [N]
sm2_FFl = np.array([462,458,454,449,465,472,482])*0.453592/(3600)       # [kg/s]
sm2_FFr = np.array([486,482,477,473,489,496,505])*0.453592/(3600)       # [kg/s]
sm2_F_used = np.array([664,694,730,755,798,825,846])*0.453592           # [kg]
sm2_temp = np.array([5.5,4.5,3.5,2.5,5.0,6.2,8.2])+273.15               # [K]
sm2_weight = (takeoff_mass - sm2_F_used) * g

# Stationary Measurements 3

sm3_time = np.array([2898,3024])                                        # [s] TODO update
sm3_alt = np.array([5730,5790])*0.514444                                # [m]
sm3_IAS = np.array([161,161])                                           # [m/s]
sm3_alpha = np.array([5.3,5.3])*np.pi/180                               # [rad]
sm3_delta_e = np.array([0,-0.5])*np.pi/180                              # [rad]
sm3_delta_tr = np.array([2.8,2.8])*np.pi/180                            # [rad]
sm3_Fe = np.array([0,-30])                                              # [N]
sm3_FFl = np.array([471,468])*0.453592/(3600)                           # [kg/s]
sm3_FFr = np.array([493,490])*0.453592/(3600)                           # [kg/s]
sm3_F_used = np.array([881,910])*0.453592                               # [kg]
sm3_temp = np.array([5,5])+273.15                                       # [K]
sm3_weight = (takeoff_mass - sm3_F_used) * g

