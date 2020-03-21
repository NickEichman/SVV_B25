import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy import stats

import Cit_par as cessna
import isa
import subprocess
import stationary_flight_data as flight_data
import weight


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

    v_e = v_t * np.sqrt(density/ isa.rho_0)
    return v_e * np.sqrt(cessna.W / weight)


def get_cm_derivatives(weight, height, v_t, delta_cg, delta_de, dd_dalpha):
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

    cm_delta = (
        -1
        / delta_de
        * averaged_weight
        / (0.5 * averaged_density * np.power(averaged_v_t, 2) * cessna.S)
        * delta_cg
        / cessna.c
    )
    cm_alpha = - dd_dalpha * cm_delta

    return cm_alpha, cm_delta


def get_reduced_elevator_deflection(de_measured, cm_delta, t_cs, t_c):
    """

    :param de_measured: Measured elevator deflection
    :param cm_delta:
    :param t_cs: Dimensionless standard thrust coefficient, for value look at Appendix F
    :param t_c: thrust coefficient
    :return: Reduced elevator deflection
    """
    return de_measured  -1 / cm_delta * cessna.Cm_tc * (t_cs - t_c)


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

    return thrust / (density * np.power(velocity, 2) * np.power(cessna.D, 2))


def plot_reduced_elevator_trim_curve(
    weight,
    height,
    thrust,
    standard_thrust,
    v_t,
    cm_delta,
    de_measured_trim
):
    """

    :param weight:
    :param height:
    :param thrust:
    :param standard_thrust: thrust from standard fuel flow (0.048 kg/s)
    :param v_t: True airspeed
    :param cm_delta: calculated value for cm_delta
    :param de_measured_trim: Measured elevator deflection from experiment with 7 data points
    :return:
    """
    reduced_velocity = get_reduced_equivalent_airspeed(weight, height, v_t)

    t_cs = get_thrust_coefficient(height, standard_thrust, reduced_velocity)
    t_c = get_thrust_coefficient(height, thrust, v_t)

    reduced_elevator_deflection = get_reduced_elevator_deflection(
        de_measured_trim, cm_delta, t_cs, t_c
    )
    v_fit = np.sort(reduced_velocity, axis=-1, kind='quicksort', order=None)
    de_fit = np.sort(reduced_elevator_deflection, axis=-1, kind='quicksort', order=None)
    fit = np.polyfit(v_fit, de_fit, 2)

    plt.plot(v_fit, fit[0]*np.power(v_fit,2)+fit[1]*v_fit+fit[2], label = "polynomial curve fit")
    plt.scatter(reduced_velocity, reduced_elevator_deflection, color = "red", label = "measurement data")
    plt.gca().invert_yaxis()
    plt.legend(loc = "best")
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
    reduced_force = f_e * cessna.W/weight

    plt.scatter(reduced_velocity, reduced_force)
    plt.gca().invert_yaxis()
    plt.show()


def run_thrust_exe(
    height, calibrated_airspeed, fuel_flow_left, fuel_flow_right, measured_temperature
):
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
    matlab_data = np.vstack(
        [height, mach_number, temperature_delta, fuel_flow_left, fuel_flow_right]
    ).T
    np.savetxt("matlab.dat", matlab_data, delimiter=" ")
    subprocess.call("thrust.exe")
    thrust = np.genfromtxt("thrust.dat")  
    thrust = thrust[:,0] + thrust[:,1]
    return thrust


def get_thrust_from_thrust_dat():
    """[summary]
    
    :return: array of thrust values for both engines for each stationary measurement read from .dat file
    """
    thrust = np.genfromtxt("thrust.dat")  
    thrust = thrust[:,0] + thrust[:,1]
    return thrust


# Stationary Measurement 1 

true_airspeed = get_true_airspeed(
    flight_data.sm1_alt, flight_data.sm1_temp, flight_data.sm1_IAS
)
cl_alpha, alpha_0, cn = get_cl_params(
    flight_data.sm1_alpha, flight_data.sm1_weight, flight_data.sm1_alt, true_airspeed
)
sm1_thrust_per_engine = np.array(
    [
        [3344.72, 3614.84],
        [2659.06, 2992.44],
        [2083.2, 2415.26],
        [1649.02, 2007.21],
        [1908.99, 2205.1],
        [1813.13, 2142.53],
    ]
)
sm1_thrust = sm1_thrust_per_engine[:,0]+sm1_thrust_per_engine[:,1]


cd_0, e, cd = get_cd_params(
    flight_data.sm1_alpha,
    flight_data.sm1_weight,
    flight_data.sm1_alt,
    true_airspeed,
    sm1_thrust,
)

# Stationary Measurement 3 

sm3_cg_arm1 = weight.fuel_to_cg(flight_data.sm3_F_used)[0]*flight_data.sm3_weight[0]
sm3_cg_arm2 = weight.fuel_to_cg(flight_data.sm3_F_used)[1]*flight_data.sm3_weight[1]
cg_shift = (- weight.mass_seat8*weight.seat8 + weight.mass_seat8*weight.seat1)*9.81
delta_cg = (- sm3_cg_arm1 + sm3_cg_arm2 + cg_shift)/(flight_data.sm3_weight[1])

#delta_cg = -0.26992 TODO get closer to this value

true_airspeed_sm3 = get_true_airspeed(
    flight_data.sm3_alt, flight_data.sm3_temp, flight_data.sm3_IAS
)
delta_de = np.diff(flight_data.sm3_delta_e)
dd_dalpha = stats.mstats.linregress(flight_data.sm2_alpha, flight_data.sm2_delta_e)[0]
cm_alpha, cm_delta = get_cm_derivatives(flight_data.sm3_weight, flight_data.sm3_alt, true_airspeed_sm3, delta_cg, delta_de, dd_dalpha)

print("cm_alpha = ", cm_alpha[0])
print("cm_delta = ", cm_delta[0])

# Stationary Measurement 2

true_airspeed_sm2 = get_true_airspeed(
    flight_data.sm2_alt, flight_data.sm2_temp, flight_data.sm2_IAS
)

sm2_thrust_per_engine = np.array([[1848.41, 2125.34],
                                [1896.41, 2178.07],
                                [1916.72, 2201.44],
                                [1959.07, 2256.36],
                                [1823.99, 2102.14],
                                [1808.84, 2066.41],
                                [1781.49, 2048.83]])

sm2_thrust = sm2_thrust_per_engine[:,0]+sm2_thrust_per_engine[:,1]

sm2_standard_thrust_per_engine = np.array([[1404.54, 1404.54],
                                [1458.9,  1458.9 ],
                                [1521.82, 1521.82],
                                [1586.89, 1586.89],
                                [1341.5,  1341.5 ],
                                [1249.86, 1249.86],
                                [1180.85, 1180.85]])
sm2_standard_thrust = sm2_standard_thrust_per_engine[:,0]+sm2_standard_thrust_per_engine[:,1]

plot_elevator_force_control_curve(flight_data.sm2_weight, flight_data.sm2_alt, true_airspeed_sm2, flight_data.sm2_Fe)
plot_reduced_elevator_trim_curve(flight_data.sm2_weight, flight_data.sm2_alt,sm2_thrust, sm2_standard_thrust, true_airspeed_sm2, cm_delta, flight_data.sm2_delta_e)

