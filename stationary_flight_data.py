import numpy as np

takeoff_mass = 14852*0.453592                                       # [kg]
masses = np.array([102,80,76,75,60,80,71,105,76])                   # [kg]
g = 9.81
# Stationary Measurements 1

sm1_time = np.array([1152,1296,1416,1577,1771,1898])                # [s]
sm1_alt = np.array([8000,7990,8000,7990,8070,8070])*0.3048          # [m]
sm1_IAS = np.array([249,217,189,162,129,119])*0.514444              # [m/s]
sm1_alpha = np.array([1.7,2.8,3.7,5.6,9.1,10.7])*np.pi/180          # [rad]
sm1_FFl = np.array([731,599,491,410,420,400])*0.453592/(3600)       # [kg/s]
sm1_FFr = np.array([770,647,538,459,458,442])*0.453592/(3600)       # [kg/s]
sm1_F_used = np.array([361,408,440,481,525,555])*0.453592           # [kg]
sm1_temp = np.array([-0.5,-3,-4.8,-6,-7.5,-8])+273.15               # [K]
sm1_weight = (takeoff_mass - sm1_F_used) * g



# Stationary Measurements 2

sm2_time = np.array([2136,2252,2328,2402,2518,2724,2799])               # [s]
sm2_alt = np.array([7940,8100,8450,8740,7810,6980,6400])*0.3048         # [m]
sm2_IAS = np.array([169,159,149,139,182,194,205])*0.514444              # [m/s]
sm2_alpha = np.array([4.9,5.6,6.5,7.6,4,3.4,2.8])*np.pi/180             # [rad]
sm2_delta_e = np.array([0.2,-0.1,-0.5,-0.9,0.7,1,1.2])*np.pi/180        # [rad]
sm2_delta_tr = np.array([3.6,3.6,3.6,3.6,3.5,3.5,3.5])*np.pi/180        # [rad]
sm2_Fe = np.array([0,-17,-34,-42,32,61,86])                             # [N]
sm2_FFl = np.array([443,441,434,430,450,463,471])*0.453592/(3600)       # [kg/s]
sm2_FFr = np.array([481,479,472,469,489,500,510])*0.453592/(3600)       # [kg/s]
sm2_F_used = np.array([627,650,671,688,717,773,791])*0.453592           # [kg]
sm2_temp = np.array([-5.5,-6.2,-7,-8.2,-5,-2.2,-1])+273.15              # [K]
sm2_weight = (takeoff_mass - sm2_F_used) * g

# Stationary Measurements 3

sm3_time = np.array([2898,3024])                                        # [s]
sm3_alt = np.array([6880,7050])*0.514444                                # [m]
sm3_IAS = np.array([168,167])                                           # [m/s]
sm3_alpha = np.array([4.8,4.9])*np.pi/180                               # [rad]
sm3_delta_e = np.array([0.3,-0.3])*np.pi/180                            # [rad]
sm3_delta_tr = np.array([3.5,3.5])*np.pi/180                            # [rad]
sm3_Fe = np.array([0,-38])                                              # [N]
sm3_FFl = np.array([457,455])*0.453592/(3600)                           # [kg/s]
sm3_FFr = np.array([495,490])*0.453592/(3600)                           # [kg/s]
sm3_F_used = np.array([820,852])*0.453592                               # [kg]
sm3_temp = np.array([-3.5,-4])+273.15                                   # [K]
sm3_weight = (takeoff_mass - sm3_F_used) * g

