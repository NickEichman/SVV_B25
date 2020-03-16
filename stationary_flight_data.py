import numpy as np

masses = np.array([80,102,76,74,58,89,71,110,87])                   # [kg]

# Stationary Measurements 1

sm1_time = np.array([1436,1533,1678,2015,2264,2338])                # [s]
sm1_alt = np.array([14990,15000,14990,14990,14840,14840])*0.3048    # [m]
sm1_IAS = np.array([238,224,188,158,125,117])*0.514444              # [m/s]
sm1_alpha = np.array([1.9,2.2,3.9,5.7,10,11.7])*np.pi/180           # [rad]
sm1_FFl = np.array([673,589,458,373,343,397])*0.453592/(3600)       # [kg/s]
sm1_FFr = np.array([719,633,498,404,369,427])*0.453592/(3600)       # [kg/s]
sm1_F_used = np.array([517,546,584,661,712,727])*0.453592/(3600)    # [kg/s]
sm1_temp = np.array([-8.2,-11.5,-14.2,-17.5,-19.8,-20.2])+273.15    # [K]


# Stationary Measurements 2

sm2_time = np.array([2562,2652,2720,2787,2918,2977,3054])               # [s]
sm2_alt = np.array([14750,14940,15160,15360,14500,14100,13510])*0.3048  # [m]
sm2_IAS = np.array([158,149,140,129,172,180,192])*0.514444              # [m/s]
sm2_alpha = np.array([5.7,6.5,7.6,9,4.6,4.1,3.4])*np.pi/180             # [rad]
sm2_delta_e = np.array([-0.2,-0.5,-1,-1.6, 0.4, 0.6, 0.9])*np.pi/180    # [rad]
sm2_delta_tr = np.array([-2.9,-2.9,-2.9,-2.9,-2.9,-2.9,-2.9])*np.pi/180 # [rad]
sm2_Fe = np.array([0,-19,-29,-38,32,49,88])                             # [N]
sm2_FFl = np.array([402,399,392,390,411,415,427])*0.453592/(3600)       # [kg/s]
sm2_FFr = np.array([430,430,422,424,444,449,462])*0.453592/(3600)       # [kg/s]
sm2_F_used = np.array([787,809,823,837,868,881,900])*0.453592/(3600)    # [kg/s]
sm2_temp = np.array([-18.2,-18.8,-19.5,-20.8,-17.8,-16.5,-15.5])+273.15 # [K]

# Stationary Measurements 3

sm3_time = np.array([3255,3371])                                        # [s]
sm3_alt = np.array([15740,13870])*0.514444                               # [m]
sm3_IAS = np.array([159,158])                                           # [m/s]
sm3_alpha = np.array([5.5,5.5])*np.pi/180                               # [rad]
sm3_delta_e = np.array([-0.1,-0.8])*np.pi/180                           # [rad]
sm3_delta_tr = np.array([-2.8,-2.8])*np.pi/180                          # [rad]
sm3_Fe = np.array([0,-30])                                              # [N]
sm3_FFl = np.array([417,414])*0.453592/(3600)                           # [kg/s]
sm3_FFr = np.array([452,450])*0.453592/(3600)                           # [kg/s]
sm3_F_used = np.array([975,979])*0.453592/(3600)                        # [kg/s]
sm3_temp = np.array([-17.5,-17.2])+273.15                               # [K]

