import numpy as np
#Input data
#Give in meters, 1 inch = 0.0254 m, 1 inchpound = 0.113 Nm, 1 pound = 0.453592 kg
bem = 9165 * 0.453592
fuel_mass = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]

nosebaggagemass =0
aftcabinmass =0
aftcabinmass2 =0
wingjackmass =0
mass_cargo =0
mass_seat1=102
mass_seat2 =80
mass_baggage =0
mass_seat3 =76
mass_seat4 =75
mass_seat5 =60
mass_seat6 =80
mass_seat7 =71
mass_seat8 =105
mass_seat10 =76
payloadmass = mass_seat1+mass_seat2+mass_seat3+mass_seat4+mass_seat5+mass_seat6+mass_seat7+mass_seat8+mass_seat10+nosebaggagemass+aftcabinmass+aftcabinmass2
ramp_mass = 14852 * 0.453592

#xcgdatum and xcg function calculations
#xcg positions of all components [INCHES]
seat1 =131 * 0.0254
seat2 =131 * 0.0254
seat3 =214 * 0.0254
seat4 =214 * 0.0254
seat5 =251 * 0.0254
seat6 =251 * 0.0254
seat7 =288 * 0.0254
seat8 =288 * 0.0254
seat10 = 170 * 0.0254
nosebaggage = 93.7 * 0.0254 #meters
aftcabbaggage =0
aftcabbaggage2 =0
bemcg =292.18 * 0.0254
payloadcg =0
wingcg = 315.5 * 0.0254
zfmcg =280.4 * 0.0254
rampmasscg = 281.76 * 0.0254

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
Cms10 = seat10*mass_seat10
Cmcab1 = aftcabbaggage*aftcabinmass
Cmcab2 = aftcabbaggage2*aftcabinmass2
Cmnose = nosebaggagemass * nosebaggage
Cmfuel = [298.16,591.18,879.08,1165.42,1448.40,1732.53,2014.80,2298.84,2581.92,2866.30,3150.18,3434.52,3718.52,4003.23,4287.76,4572.24,4856.56,5141.16,5425.64,5709.9,5994.04,6278.47,6562.82,6846.96,7131.00,7415.33,7699.60,7984.34,8269.06,8554.05,8839.04,9124.80,9410.62,9696.97,9983.40,10270.08,10556.84,10843.87,11131.00,11418.20]
print(np.interp([2000,3000,4000], fuel_mass, Cmfuel))

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
            cmtotal.append(Cmramp + Cms1 + Cms2 + Cms3 + Cms4 + Cms5 + Cms6 + Cms7 + Cms8 + Cms10 + Cmcab1 + Cmcab2 + Cmnose +cmfuelcurrent) #Current aircraft Cm
            xcg.append(cmtotal[-1] / weights[-1])
    return area,areaq1,weights, cmtotal, xcg
print(Cmramp, Cms1, Cms2, Cms3, Cms4, Cms5, Cms6, Cms7, Cms8, Cms10)


