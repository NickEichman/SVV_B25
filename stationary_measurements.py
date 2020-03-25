import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from scipy import stats

import Cit_par as cessna
import isa
import subprocess
import stationary_flight_data as flight_data
import weight


def get_r_2(y, y_fit):
    mean = np.mean(y)
    ss_tot = np.sum(np.power(y - mean, 2))
    ss_reg = np.sum(np.power(y_fit - mean, 2))
    return ss_reg / ss_tot


def cl_curve(alpha, cn_alpha, alpha_0):
    return cn_alpha * alpha - cn_alpha * alpha_0


def cd_curve(cl, cd_0, e):
    return cd_0 + np.power(cl, 2) / (np.pi * cessna.A * e)


def plot_cl_alpha_curve(alpha, cn, cn_alpha, alpha_0):
    y_fit = cl_curve(alpha, cn_alpha, alpha_0)
    y = cn

    r_2 = get_r_2(y, y_fit)
    plt.plot(
        np.degrees(alpha), y_fit, label=r"Linear fit $r^{2}$= " + str(round(r_2, 4)),
    )
    plt.scatter(
        np.degrees(alpha), y, color="red", label="Measured data",
    )
    plt.grid()
    plt.ylabel(r"$C_{l}$ $[-]$", fontsize="x-large")
    plt.xlabel(r"$\alpha$ $[^{\circ}]$", fontsize="x-large")
    plt.legend(loc="best")
    plt.show()


def plot_cd_alpha_curve(alpha, cn):
    x = alpha
    y = cd_curve(cn, cd_0, e)

    fit = np.polyfit(x, y, 2)

    x_fit = np.linspace(x[0], x[-1])
    y_fit = fit[0] * np.power(x_fit, 2) + fit[1] * x_fit + fit[2]

    r_2 = get_r_2(y, fit[0] * np.power(x, 2) + fit[1] * x + fit[2])
    plt.plot(
        np.degrees(x_fit),
        y_fit,
        label=r"Second order polynomial fit $r^{2}$= " + str(round(r_2, 4)),
    )
    plt.scatter(
        np.degrees(x), y, color="red", label="Measured data",
    )
    plt.grid()
    plt.ylabel(r"$C_{d}$ $[-]$", fontsize="x-large")
    plt.xlabel(r"$\alpha$ $[^{\circ}]$", fontsize="x-large")
    plt.legend(loc="best")
    plt.show()


def plot_cl_cd_curve(alpha, cn, cn_alpha, alpha_0):

    y = cl_curve(alpha, cn_alpha, alpha_0)
    x = cd_curve(cn, cd_0, e)

    fit = np.polyfit(x, y, 2)

    x_fit = np.linspace(x[0], x[-1])
    y_fit = fit[0] * np.power(x_fit, 2) + fit[1] * x_fit + fit[2]

    r_2 = get_r_2(y, fit[0] * np.power(x, 2) + fit[1] * x + fit[2])
    plt.plot(
        x_fit,
        y_fit,
        label=r"Second order polynomial fit $r^{2}$= " + str(round(r_2, 4)),
    )
    plt.scatter(
        x, y, color="red", label="Measured data",
    )
    plt.grid()
    plt.ylabel(r"$C_{l}$ $[-]$", fontsize="x-large")
    plt.xlabel(r"$C_{d}$ $[-]$", fontsize="x-large")
    plt.legend(loc="best")
    plt.show()


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

    v_e = v_t * np.sqrt(density / isa.rho_0)
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
    cm_alpha = -dd_dalpha * cm_delta

    return cm_alpha, cm_delta


def get_reduced_elevator_deflection(de_measured, cm_delta, t_cs, t_c):
    """

    :param de_measured: Measured elevator deflection
    :param cm_delta:
    :param t_cs: Dimensionless standard thrust coefficient, for value look at Appendix F
    :param t_c: thrust coefficient
    :return: Reduced elevator deflection
    """
    return de_measured - 1 / cm_delta * cessna.Cm_tc * (t_cs - t_c)


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
    weight, height, thrust, standard_thrust, v_t, cm_delta, de_measured_trim, test=False
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
    v_fit = np.sort(reduced_velocity, axis=-1, kind="quicksort", order=None)
    de_fit = np.sort(reduced_elevator_deflection, axis=-1, kind="quicksort", order=None)
    fit = np.polyfit(v_fit, de_fit, 2)

    r_2 = get_r_2(de_fit, fit[0] * np.power(v_fit, 2) + fit[1] * v_fit + fit[2])

    if test == False:
        plt.plot(
            v_fit,
            fit[0] * np.power(v_fit, 2) + fit[1] * v_fit + fit[2],
            label=r"Second order polynomial fit $r^{2}$= " + str(round(r_2, 4)),
        )
        plt.scatter(
            reduced_velocity,
            reduced_elevator_deflection,
            color="red",
            label="Measured data",
        )
    else:
        plt.plot(
            v_fit,
            fit[0] * np.power(v_fit, 2) + fit[1] * v_fit + fit[2],
            label=r"cm_delta = " + str(round(cm_delta[0], 3)),
        )

    plt.grid()
    plt.gca().invert_yaxis()
    plt.ylabel(r"$-\delta^{*}_{eq}$ $[rad]$", fontsize="x-large")
    plt.xlabel(r"$\tilde{V}_{eq}$ $[\frac{m}{s}]$", fontsize="x-large")
    plt.legend(loc="best")
    if test == False:
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
    reduced_force = f_e * cessna.W / weight

    v_fit = np.sort(reduced_velocity, axis=-1, kind="quicksort", order=None)
    de_fit = np.sort(reduced_force, axis=-1, kind="quicksort", order=None)
    fit = np.polyfit(v_fit, de_fit, 2)

    r_2 = get_r_2(de_fit, fit[0] * np.power(v_fit, 2) + fit[1] * v_fit + fit[2])

    plt.plot(
        v_fit,
        fit[0] * np.power(v_fit, 2) + fit[1] * v_fit + fit[2],
        label=r"Second order polynomial fit $r^{2}$= " + str(round(r_2, 4)),
    )
    plt.grid()
    plt.scatter(reduced_velocity, reduced_force, color="red", label="Measured data")
    plt.gca().invert_yaxis()
    plt.ylabel(r"$-F^{*}_{e}$ $[rad]$", fontsize="x-large")
    plt.xlabel(r"$\tilde{V}_{eq}$ $[\frac{m}{s}]$", fontsize="x-large")
    plt.legend(loc="best")

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
    thrust = thrust[:, 0] + thrust[:, 1]
    return thrust


def get_thrust_from_thrust_dat():
    """[summary]
    
    :return: array of thrust values for both engines for each stationary measurement read from .dat file
    """
    thrust = np.genfromtxt("thrust.dat")
    thrust = thrust[:, 0] + thrust[:, 1]
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
sm1_thrust = sm1_thrust_per_engine[:, 0] + sm1_thrust_per_engine[:, 1]


cd_0, e, cd = get_cd_params(
    flight_data.sm1_alpha,
    flight_data.sm1_weight,
    flight_data.sm1_alt,
    true_airspeed,
    sm1_thrust,
)

print("cl_alpha = ", round(cl_alpha, 3))
print("cd_0 = ", round(cd_0, 3))
print("e = ", round(e, 3))
print("alpha_0 = ", round(alpha_0, 3))

# Stationary Measurement 3

sm3_cg_arm1 = weight.fuel_to_cg(flight_data.sm3_F_used)[0] * flight_data.sm3_weight[0]
sm3_cg_arm2 = weight.fuel_to_cg(flight_data.sm3_F_used)[1] * flight_data.sm3_weight[1]
cg_shift = (-weight.mass_seat8 * weight.seat8 + weight.mass_seat8 * weight.seat1) * 9.81
delta_cg = (-sm3_cg_arm1 + sm3_cg_arm2 + cg_shift) / (flight_data.sm3_weight[1])

# delta_cg = -0.26992 TODO get closer to this value

true_airspeed_sm3 = get_true_airspeed(
    flight_data.sm3_alt, flight_data.sm3_temp, flight_data.sm3_IAS
)
delta_de = np.diff(flight_data.sm3_delta_e)
dd_dalpha = stats.mstats.linregress(flight_data.sm2_alpha, flight_data.sm2_delta_e)[0]
cm_alpha, cm_delta = get_cm_derivatives(
    flight_data.sm3_weight,
    flight_data.sm3_alt,
    true_airspeed_sm3,
    delta_cg,
    delta_de,
    dd_dalpha,
)
print("cm_alpha = ", cm_alpha)
print("cm_delta = ", cm_delta)

# Stationary Measurement 2

true_airspeed_sm2 = get_true_airspeed(
    flight_data.sm2_alt, flight_data.sm2_temp, flight_data.sm2_IAS
)

sm2_thrust_per_engine = np.array(
    [
        [1848.41, 2125.34],
        [1896.41, 2178.07],
        [1916.72, 2201.44],
        [1959.07, 2256.36],
        [1823.99, 2102.14],
        [1808.84, 2066.41],
        [1781.49, 2048.83],
    ]
)

sm2_thrust = sm2_thrust_per_engine[:, 0] + sm2_thrust_per_engine[:, 1]

sm2_standard_thrust_per_engine = np.array(
    [
        [1404.54, 1404.54],
        [1458.9, 1458.9],
        [1521.82, 1521.82],
        [1586.89, 1586.89],
        [1341.5, 1341.5],
        [1249.86, 1249.86],
        [1180.85, 1180.85],
    ]
)
sm2_standard_thrust = (
    sm2_standard_thrust_per_engine[:, 0] + sm2_standard_thrust_per_engine[:, 1]
)


# Plotting

plot_elevator_force_control_curve(
    flight_data.sm2_weight, flight_data.sm2_alt, true_airspeed_sm2, flight_data.sm2_Fe
)
plot_reduced_elevator_trim_curve(
    flight_data.sm2_weight,
    flight_data.sm2_alt,
    sm2_thrust,
    sm2_standard_thrust,
    true_airspeed_sm2,
    cm_delta,
    flight_data.sm2_delta_e,
)
plot_cl_alpha_curve(flight_data.sm1_alpha, cn, cl_alpha, alpha_0)
plot_cl_cd_curve(flight_data.sm1_alpha, cn, cl_alpha, alpha_0)
plot_cd_alpha_curve(flight_data.sm1_alpha, cn)


if __name__ == "__main__":
    """
    Test 1:
    Moving cg aft reduces slope of cm_alpha. Test 1 assesses impact on trim curves when moving cg aft 
    The elevator deflection is also reduced accordingly. It is ecpected that initial magnitude and slope reduce 
    """
    cm_alpha = np.array(
        [cm_alpha, cm_alpha * 0.8, cm_alpha * 0.4, cm_alpha * 0.2, cm_alpha * 0.1]
    )
    cm_delta = -cm_alpha / dd_dalpha
    delta_e = np.array(
        [
            flight_data.sm2_delta_e,
            flight_data.sm2_delta_e * 0.8,
            flight_data.sm2_delta_e * 0.4,
            flight_data.sm2_delta_e * 0.2,
            flight_data.sm2_delta_e * 0.1,
        ]
    )

    for i in range(len(cm_alpha)):

        plot_reduced_elevator_trim_curve(
            flight_data.sm2_weight,
            flight_data.sm2_alt,
            sm2_thrust,
            sm2_standard_thrust,
            true_airspeed_sm2,
            cm_delta[i],
            delta_e[i],
            True,
        )

    plt.show()

    """
    Test 2:
    Increase Fuel used and assess impact on cm_alpha and cm_delta. Both should become less negative as cg moves aft 
    Reducing ampunt of fuel moves cg forward
    """
    cg1 = weight.fuel_to_cg(flight_data.sm3_F_used)[0]

    flight_data.sm3_F_used *= 1.5

    cg2 = weight.fuel_to_cg(flight_data.sm3_F_used)[0]

    sm3_cg_arm1 = (
        weight.fuel_to_cg(flight_data.sm3_F_used)[0] * flight_data.sm3_weight[0]
    )
    sm3_cg_arm2 = (
        weight.fuel_to_cg(flight_data.sm3_F_used)[1] * flight_data.sm3_weight[1]
    )
    cg_shift = (
        -weight.mass_seat8 * weight.seat8 + weight.mass_seat8 * weight.seat1
    ) * 9.81
    delta_cg = (-sm3_cg_arm1 + sm3_cg_arm2 + cg_shift) / (flight_data.sm3_weight[1])

    cm_alpha_new, cm_delta_new = get_cm_derivatives(
        flight_data.sm3_weight,
        flight_data.sm3_alt,
        true_airspeed_sm3,
        delta_cg,
        delta_de,
        dd_dalpha,
    )

    print("change in cg = ", round(cg1 - cg2, 3), " m")
    print("modified cm_alpha = ", cm_alpha_new)
    print("modified cm_delta = ", cm_delta_new)

    assert (
        cm_alpha_new < cm_alpha[0]
    ), "cm_alpha is not reducing with a foward shift in cg"

    """
    Test 3:
    Assess if calculated Mach number is equal to Mach number calculated by hand (0.312) for 100 m/s CAS and 1000m height.
    """
    np.testing.assert_almost_equal(
        get_mach_number(1000, 100), 0.312, 3, "Incorrect conversion to Mach number"
    )

    """
    Test 4:
    Assess if true airspeed is equal to true airspeed calculated by hand (104.836) for 1000m height, 287.12 measured tempearture and 100 m/s CAS
    """
    np.testing.assert_almost_equal(
        get_true_airspeed(1000, 287.12, 100),
        104.836,
        2,
        "Incorrect conversion to True Airspeed",
    )

    """
    Test 5:
    Assess if reduced equivalent airspeed is equal to reduced equivalent airspeed calculated by hand(776.886) with weight of 1000N, height of 1000m and 104.82622 TAS.
    """
    np.testing.assert_almost_equal(
        get_reduced_equivalent_airspeed(1000, 1000, 104.82622864721496),
        776.88,
        2,
        "Incorrect conversion to reduced Equivalent Airspeed",
    )

    """
    Test 6:
    Assess if R^2 function will yield 1 in case of perfect fit
    """
    np.testing.assert_almost_equal(
        get_r_2(np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3, 4, 5, 6])),
        1,
        err_msg="R2 value not equal to 1 for perfect fir",
    )
