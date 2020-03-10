import numpy as np
#Input data
#Give in meters, 1 inch = 0.0254 m, 1 inchpound = 0.113 Nm
bem = 1
fuel_mass = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5008]

nosebaggagemass =1
aftcabinmass =1
aftcabinmass2 =1
wingjackmass =1
mass_cargo =1
mass_seat1=1
mass_seat2 =1
mass_baggage =1
mass_seat3 =1
mass_seat4 =1
mass_seat5 =1
mass_seat6 =1
mass_seat7 =1
mass_seat8 =1
mass_seat9 =1
mass_seat10 =1
payloadmass = mass_seat1+mass_seat2+mass_seat3+mass_seat4+mass_seat5+mass_seat6+mass_seat7+mass_seat8+mass_seat10+mass_seat9+nosebaggagemass+aftcabinmass+aftcabinmass2
#mass calculations
ramp_mass = bem + fuel_mass[-1] +mass_baggage+mass_cargo+mass_seat1+mass_seat2+mass_seat3+mass_seat4+mass_seat5+mass_seat6+mass_seat7+mass_seat8+mass_seat9+mass_seat10

#xcgdatum and xcg function calculations
#xcg positions of all components [INCHES]
seat1 =1
seat2 =1
seat3 =1
seat4 =1
seat5 =1
seat6 =1
seat7 =1
seat8 =1
seat9 =1
seat10 =1
nosebaggage = 93.7 * 0.0254 #meters
aftcabbaggage =1
aftcabbaggage2 =1
bemcg =1
payloadcg =1
wingcg = 315.5 * 0.0254
zfmcg =1
rampmasscg = 300.21 - 218.2 * nosebaggagemass/(wingjackmass+nosebaggagemass)

#moments of all components
Cmramp = ramp_mass*rampmasscg
Cms1 = seat1 * mass_seat1
Cms2 = seat2 * mass_seat2
Cms3 = seat3 * mass_seat3
Cms4 = seat4 * mass_seat4
Cms5 = seat5 * mass_seat5
Cms6 = seat6 * mass_seat6
Cms7 = seat7 * mass_seat7
Cms8 = seat8 * mass_seat8
Cms9 = seat9*mass_seat9
Cms10 = seat10*mass_seat10
Cmcab1 = aftcabbaggage*aftcabinmass
Cmcab2 = aftcabbaggage2*aftcabinmass2
Cmnose = nosebaggagemass * nosebaggage
Cmfuel = [298.16,591.18,879.08,1165.42,1448.40,1732.53,2014.80,2298.84,2581.92,2866.30,3150.18,3434.52,3718.52,4003.23,4287.76,4572.24,4856.56,5141.16,5425.64,5709.9,5994.04,6278.47,6562.82,6846.96,7131.00,7415.33,7699.60,7984.34,8269.06,8554.05,8839.04,9124.80,9410.62,9696.97,9983.40,10270.08,10556.84,10843.87,11131.00,11418.20,11705.50,11993.31,12281.18,12569.04,12856.86,13144.73,13432.48,13720.56,14008.46,14320.34]
print(np.interp([150,250,350], fuel_mass, Cmfuel))

#trapezoidal for fuel consumption
#paste lists of fuel flow and time
def trapezodial(x,y):
    areaq1= []
    weights = []
    cmtotal = []
    xcg = []
    if len(x)!=len(y):
        print("Length of the two list are not equal.")
    area = 0
    for i in range(len(x)-1):
        area += (x[i+1]-x[i])/2 * (y[i+1]+y[i])
        areaq1.append(area)  #Fuel used
        for i in areaq1:
            weights.append = ramp_mass - areaq1[-1] #Current aircraft weight
            cmfuelcurrent = np.interp((fuel_mass[-1]-areaq1[-1]), fuel_mass, Cmfuel) #Interpolated value for cm_fuel
            cmtotal.append(Cmramp + Cms1 + Cms2 + Cms3 + Cms4 + Cms5 + Cms6 + Cms7 + Cms8 + Cms10 + Cmcab1 + Cmcab2 + Cmnose +cmfuelcurrent + Cms9) #Current aircraft Cm
            xcg.append(cmtotal[-1] / weights[-1])
    return area,areaq1,weights, cmtotal, xcg

#weight = ramp_mass - trapezodial(totalfuelflow, time)
#weight = ramp_mass - (fuel_mass[-1]-fuel_mass[-2])
