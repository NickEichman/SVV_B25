import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

import Cit_par as cessna
import isa
import subprocess
import stationary_flight_data as flight_data

def cl_curve(alpha, cn_alpha, alpha_0):
    return cn_alpha * alpha - cn_alpha * alpha_0


def cd_curve(cl, cd_0, e):
    return cd_0 + np.power(cl, 2) / (np.pi * cessna.A * e)


def get_cl_params(alpha, weight, height, true_airspeed):
    """
    Inputs should be single values or numpy arrays

    :param alpha: angle of attack
    :param weight:
    :param height:
    :param true_airspeed:
    :return: cn_alpha, alpha_0, cn from flight
    """

    temperature = isa.get_t_at_height(height)
    pressure = isa.get_p_at_temperature(temperature)
    density = isa.get_rho(pressure, temperature)
    cn = weight / (0.5 * density * np.power(true_airspeed, 2) * cessna.S)

    popt, _ = opt.curve_fit(cl_curve, alpha, cn)
    cn_alpha, alpha_0 = popt

    return cn_alpha, alpha_0, cn


def get_cd_params(alpha, weight, height, true_airspeed, thrust):
    """
    Inputs should be single values or numpy arrays

    :param alpha: angle of attack
    :param weight:
    :param height:
    :param true_airspeed:
    :param thrust:
    :return: cd_0, e, cd from flight
    """
    temperature = isa.get_t_at_height(height)
    pressure = isa.get_p_at_temperature(temperature)
    density = isa.get_rho(pressure, temperature)

    cn_alpha, alpha_0, _ = get_cl_params(alpha, weight, height, true_airspeed)

    cl = cn_alpha * (alpha - alpha_0)
    cd = thrust / (0.5 * density * np.power(true_airspeed, 2) * cessna.S)

    popt, _ = opt.curve_fit(cd_curve, cl, cd)
    cd_0, e = popt
    return cd_0, e, cd


def get_mach_number(height, v_c):
    """

    :param height:
    :param v_c: Calibrated airspeed
    :return: Mach number
    """
    pressure = isa.get_p_at_temperature(isa.get_t_at_height(height))

    innermost = np.power(
        1
        + (isa.gamma - 1) / (2 * isa.gamma) * (isa.rho_0 / isa.p_0) * np.power(v_c, 2),
        isa.gamma / (isa.gamma - 1),
    )
    middle = np.power(
        1 + isa.p_0 / pressure * (innermost - 1), (isa.gamma - 1) / isa.gamma
    )
    M = np.sqrt(2 / (isa.gamma - 1) * (middle - 1))

    return M


def get_true_airspeed(height, measured_temperature, v_c):
    """

    :param height:
    :param measured_temperature:
    :param v_c: Calibrated airspeed
    :return: True airspeed
    """
    M = get_mach_number(height, v_c)
    temperature = measured_temperature / (1 + (isa.gamma - 1) / 2 * np.power(M, 2))
    a = isa.get_speed_of_sound(temperature)

    return M * a

def get_reduced_equivalent_airspeed(weight, height, v_t):
    """

    :param weight:
    :param height:
    :param v_t: True airspeed
    :return: Reduced equivalent airspeed
    """
    temperature = isa.get_t_at_height(height)
    pressure = isa.get_p_at_temperature(temperature)
    density = isa.get_rho(pressure, temperature)

    v_e = v_t*np.sqrt(density, isa.rho_0)
    return  v_e*np.sqrt(cessna.W/weight)

def get_cm_derivatives(weight, height, v_t, delta_cg, delta_de, delta_alpha):
    """

    :param weight:
    :param height:
    :param v_t: True airspeed
    :param delta_cg: Change in CG location
    :param delta_de: Change in elevator deflection
    :param delta_alpha: Change in angle of attack
    :return: cm_alpha, cm_delta
    """
    averaged_weight = np.mean(weight)
    temperature = isa.get_t_at_height(height)
    pressure = isa.get_p_at_temperature(temperature)
    density = isa.get_rho(pressure, temperature)
    averaged_density = np.mean(density)
    averaged_v_t = np.mean(v_t)

    cm_delta = -1/delta_de*averaged_weight/(0.5*averaged_density*np.power(averaged_v_t,2)*cessna.S)*delta_cg/cessna.c
    cm_alpha = -delta_de/delta_alpha*cm_delta

    return cm_alpha, cm_delta

def get_reduced_elevator_deflection(de_measured, cm_delta, t_cs, t_c):
    """

    :param de_measured: Measured elevator deflection
    :param cm_delta:
    :param t_cs: Dimensionless standard thrust coefficient, for value look at Appendix F
    :param t_c: thrust coefficient
    :return: Reduced elevator deflection
    """
    return de_measured* -1/cm_delta*cessna.Cm_tc*(t_cs-t_c)

def get_thrust_coefficient(height, thrust, velocity):
    """

    :param height:
    :param thrust:
    :param rps: Revolutions per second
    :return: Thrust coefficient
    """
    temperature = isa.get_t_at_height(height)
    pressure = isa.get_p_at_temperature(temperature)
    density = isa.get_rho(pressure, temperature)

    return thrust/(density*np.power(velocity,2)*np.power(cessna.D,2))

def plot_reduced_elevator_trim_curve(weight, height, thrust, thrust_from_exe, v_t, de_measured_cg, alpha_measured_cg, de_measured_trim,delta_cg):
    """

    :param weight:
    :param height:
    :param thrust:
    :param v_t: True airspeed
    :param de_measured_cg: Measured elevator deflection from experiment with change of cg
    :param de_measured_trim: Measured elevator deflection from experiment with 7 data points
    :param alpha_measured_cg: Measured angle of attack
    :param delta_cg: Change in center of gravity
    :return:
    """
    reduced_velocity = get_reduced_equivalent_airspeed(weight, height, v_t)

    delta_de = np.diff(de_measured_cg)
    delta_alpha = np.diff(alpha_measured_cg)
    t_cs = get_thrust_coefficient(height, thrust_from_exe, reduced_velocity)
    t_c = get_thrust_coefficient(height, thrust, v_t)

    _, cm_delta = get_cm_derivatives(weight, height, v_t, delta_cg, delta_de,delta_alpha)
    reduced_elevator_deflection = get_reduced_elevator_deflection(de_measured_trim, cm_delta, t_cs, t_c)


    plt.plot(reduced_velocity, reduced_elevator_deflection)
    plt.show()

def plot_elevator_force_control_curve(weight, height, v_t, f_e):
    """

    :param weight:
    :param height:
    :param v_t:
    :param f_e: Stick force
    :return:
    """
    reduced_velocity = get_reduced_equivalent_airspeed(weight, height, v_t)
    reduced_force = f_e*weight/cessna.W

    plt.plot(reduced_velocity, reduced_force)
    plt.show()


def run_thrust_exe(height, calibrated_airspeed, fuel_flow_left, fuel_flow_right, measured_temperature):
    """[summary]
    
    :param height: from stationary measurements 
    :param calibrated_airspeed: from stationary measurements 
    :param fuel_flow_left: from stationary measurements 
    :param fuel_flow_right: from stationary measurements 
    :param measured_temperature: from stationary measurements 
    :return: array of thrust values for both engines for each stationary measurement from thrust.exe program 
    """    
    mach_number = get_mach_number(height, calibrated_airspeed)
    temperature_delta = measured_temperature - isa.get_t_at_height(height)
    matlab_data = np.vstack([height, mach_number, fuel_flow_left, fuel_flow_right, temperature_delta]).T
    np.savetxt("matlab.dat", matlab_data, delimiter=" ")
    subprocess.call("thrust.exe")
    thrust = np.genfromtxt("thrust.dat")[:,0]*2 #per engine, therfore x2
    return thrust

def get_thrust_from_thrust_dat():
    """[summary]
    
    :return: array of thrust values for both engines for each stationary measurement read from .dat file
    """    
    thrust = np.genfromtxt("thrust.dat")[:, 0] * 2  # per engine, therfore x2
    return thrust

true_airspeed = get_true_airspeed(flight_data.sm1_alt, flight_data.sm1_temp, flight_data.sm1_IAS)
cl_alpha, alpha_0, cn = get_cl_params(flight_data.sm1_alpha, flight_data.sm1_weight, flight_data.sm1_alt,true_airspeed)
sm1_thrust =np.array([7259.3,  6334.22, 4817.2,  3784.28, 3602.42, 4594.08]) #TODO CHANGE WHEN DATA IS CHANGED

cd_0, e, cd = get_cd_params(flight_data.sm1_alpha, flight_data.sm1_weight, flight_data.sm1_alt, true_airspeed, sm1_thrust)