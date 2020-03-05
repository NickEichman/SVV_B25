import numpy as np
import math

#PHYSICAL CONSTANTS, SI UNITS
T_0 = 288.15
alpha = -6.5e-3
R = 287
P_0 = 101325
gamma = 1.4
g = 9.80665

def get_rho(p, t, R=R):
    """
    Calculates density for given pressure, valid for stratosphere only

    :param p: Pressure [Pa]
    :param t: Temperature [K]
    :param R: Gas constant of the air
    :return: Density [kg/m^3]
    """
    return p/(R*t)

def get_t_at_height(h, alpha=alpha, T_0=T_0):
    """
    Calculates temperature for given height, valid for stratosphere only

    :param h: height [m]
    :param alpha: constant
    :param T_0: ground temperature [K]
    :return: Temperature at height h [K]
    """
    return T_0 + alpha*h

def get_p_at_temperature(t, P_0=P_0, T_0=T_0, g=g, alpha=alpha, R=R):
    """
    Calculates pressure for given temperature, valid for stratosphere only

    :param t: temperature [K]
    :param P_0: ground pressure [Pa]
    :param T_0: ground temperature [K]
    :param g: gravitational acceleration [m/s^2]
    :param alpha: constant
    :param R: Gas constant of the air
    :return: Pressure at temperature t [Pa]
    """
    return P_0*np.power(t/T_0, g/(-alpha*R))

def get_speed_of_sound(t, gamma=gamma, R=R):
    """
    Calculates speed of sound for given temperature, valid for stratosphere only

    :param t: temperature [K]
    :param gamma: constant
    :param R: Gas constant of the air
    :return: Speed of sound at temperature T[K]
    """
    return math.sqrt(gamma*R*t)

def get_h_at_pressure(p, T_0=T_0, P_0=P_0, alpha=alpha, g=g, R=R):
    """
    Calculates height for given pressure, valid for stratosphere only

    :param p: pressure [Pa]
    :param T_0: ground temperature [K]
    :param P_0: ground pressure [Pa]
    :param alpha: constant
    :param g: gravitational acceleration [m/s^2]
    :param R: gas constant of the air
    :return: height [m]
    """
    pressure_ratio = p/P_0
    exponent = -alpha*R/g
    return T_0*(np.power(pressure_ratio, exponent)-1)/alpha

if __name__=="__main__":

    """
    Test case 1:
    Pressure at height 1000m should be equal to 89872.578 Pa
    """
    np.testing.assert_array_almost_equal(get_p_at_temperature(get_t_at_height(1000)), 89872.578, 3, "incorrect pressure at altitude 1000m")

    """
    Test case 2:
    Height at pressure 89872.578 Pa should be equal to 1000 m
    """
    np.testing.assert_array_almost_equal(get_h_at_pressure(89872.578), 1000, 3, "incorrect height for pressure 89872.578 Pa")
