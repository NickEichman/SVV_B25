import numpy as np

# Citation 550 - Linear simulation

# xcg = 0.25 * c

# Stationary flight condition
class Cit_par:
    def __init__(self,aoa0,height0,Vel0,theta0):
        self.hp0 = height0 # pressure altitude in the stationary flight condition [m]
        self.V0 = Vel0 # true airspeed in the stationary flight condition [m/sec]
        self.alpha0 = aoa0  # angle of attack in the stationary flight condition [rad]
        self.th0 = theta0  # pitch angle in the stationary flight condition [rad]

        # Aircraft mass
        self.m = 13600*0.453592  # mass [kg]

        # aerodynamic properties
        self.e = 0.865326  # Oswald factor [ ]
        self.CD0 = 0.02096  # Zero lift drag coefficient [ ]
        self.CLa = 4.7004  # Slope of CL-alpha curve [ ]

        # Longitudinal stability

        self.Cma = -0.63388 # longitudinal stabilty [ ]
        self.Cmde = -1.39711   # elevator effectiveness [ ]

        # Aircraft geometry

        self.S = 30.00  # wing area [m^2]
        self.Sh = 0.2 * self.S  # stabiliser area [m^2]
        self.Sh_S = self.Sh / self.S  # [ ]
        self.lh = 0.71 * 5.968  # tail length [m]
        self.c = 2.0569  # mean aerodynamic cord [m]
        self.lh_c = self.lh / self.c  # [ ]
        self.b = 15.911  # wing span [m]
        self.bh = 5.791  # stabilser span [m]
        self.A = self.b ** 2 / self.S  # wing aspect ratio [ ]
        self.Ah = self.bh ** 2 / self.Sh  # stabilser aspect ratio [ ]
        self.Vh_V = 1  # [ ]
        self.ih = -2 * np.pi / 180  # stabiliser angle of incidence [rad]

        # Constant values concerning atmosphere and gravity

        rho0 = 1.2250  # air density at sea level [kg/m^3]
        lamb = -0.0065  # temperature gradient in ISA [K/m]
        Temp0 = 288.15  # temperature at sea level in ISA [K]
        R = 287.05  # specific gas constant [m^2/sec^2K]
        g = 9.81  # [m/sec^2] (gravity constant)

        # air density [kg/m^3]
        self.rho = rho0 * np.power(((1 + (lamb * self.hp0 / Temp0))), (-((g / (lamb * R)) + 1)))
        self.W = self.m * g  # [N]       (aircraft weight)

        # Constant values concerning aircraft inertia

        self.muc = self.m / (self.rho * self.S * self.c)
        self.mub = self.m / (self.rho * self.S * self.b)
        self.KX2 = 0.019
        self.KZ2 = 0.042
        self.KXZ = 0.002
        self.KY2 = 1.25 * 1.114

        # Aerodynamic constants

        self.Cmac = 0  # Moment coefficient about the aerodynamic centre [ ]
        self.CNwa = self.CLa  # Wing normal force slope [ ]
        self.CNha = 2 * np.pi * self.Ah / (self.Ah + 2)  # Stabiliser normal force slope [ ]
        self.depsda = 4 / (self.A + 2)  # Downwash gradient [ ]

        # Lift and drag coefficient

        self.CL = 2 * self.W / (self.rho * self.V0 ** 2 * self.S)  # Lift coefficient [ ]
        self.CD = self.CD0 + (self.CLa * self.alpha0) ** 2 / (np.pi * self.A * self.e)  # Drag coefficient [ ]

        # Stabiblity derivatives

        self.CX0 = self.W * np.sin(self.th0) / (0.5 * self.rho * self.V0 ** 2 * self.S)
        print(self.CX0)
        self.CXu = -0.02792*2.6#both large
        self.CXa = +0.47966# Positive! (has been erroneously negative since 1993) #both small
        self.CXadot = +0.08330 #no effect
        self.CXq = -0.28170 #no effect
        self.CXde = -0.03728 #no effect

        self.CZ0 = -self.W * np.cos(self.th0) / (0.5 * self.rho * self.V0 ** 2 * self.S)
        print(self.CZ0)
        self.CZu = -0.376160#large diffrence
        self.CZa = -5.74340
        self.CZadot = -0.00350 #factor 100
        self.CZq = -5.66290
        self.CZde = -0.69612
 

        self.Cmu = +0.06990
        self.Cmadot = +0.17800
        self.Cmq = -8.79415

        self.CYb = -0.7500
        self.CYbdot = 0
        self.CYp = -0.0304
        self.CYr = +0.8495
        self.CYda = -0.0400
        self.CYdr = +0.2300

        self.Clb = -0.10260*1.2 #dampening
        self.Clp = -0.71085*0.6 #scale yaw
        self.Clr = +0.23760*0.1 #light frequence
        self.Clda = -0.23088 #roll offset
        self.Cldr = +0.03440 #initial peak

        self.Cnb = +0.1348*0.12#amplitude
        self.Cnbdot = 0
        self.Cnp = -0.0602*0.4
        self.Cnr = -0.2061*0.4 #frequency roll
        self.Cnda = -0.0120*0.1
        self.Cndr = -0.0939*0.3 #scale both

        #Pitch moment derivatives
        self.Cm_tc = -0.0064

        #Engine characteristic diameter
        self.D = 0.5
