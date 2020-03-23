#import control as ct
import numpy as np
import statespace_ref as st
import matplotlib.pyplot as plt
import scipy.signal as ct
import Cit_par_ref as cp
import flight_data_ref as fd

sym =  True
a_sym = False

if sym:
    
    data,T = fd.get_data_eigen(0)
    aoa = np.transpose(np.array(data[0]))[0]*np.pi/180
    tas = np.transpose(np.array(data[1]))[0]
    pitch = np.transpose(np.array(data[3]))[0]*np.pi/180
    pitch_rate = np.transpose(np.array(data[6]))[0]*np.pi/180
    uv = np.transpose(np.array(data[9]))[0]*np.pi/180
    hp = np.transpose(np.array(data[11]))[0]*np.pi/180
    cs = cp.Cit_par(aoa[0],hp[0],tas[0],pitch[0])

    A_s,B_s,C_s,D_s = st.get_ss_symmetric(aoa[0],hp[0],tas[0],pitch[0])
    sys_s = ct.lti(A_s,B_s,C_s,D_s)
    """
    plt.plot(aoa)
    plt.plot(uv)
    plt.show()"""

    t1, y1, x1  = ct.lsim(sys_s,uv,T,[0,aoa[0],pitch[0],pitch_rate[0]])
    u=cs.V0*(y1[:,0]+1)
    alpha = y1[:,1]
    theta = y1[:,2]
    q = y1[:,3]*cs.V0/cs.c

    eig_s = np.linalg.eig(A_s)[0]
    T12 = -0.693/(np.real(eig_s))*cs.c/cs.V0
    omega_0 = np.sqrt(np.real(eig_s)**2+np.imag(eig_s)**2)*cs.V0/cs.c
    P = 2*np.pi/np.imag(eig_s)*cs.c/cs.V0
    damp = -np.real(eig_s)/np.sqrt(np.real(eig_s)**2+np.imag(eig_s)**2)

    print("A matrix for symmetric:")
    print(A_s)
    print("B matrix for symmetric:")
    print(B_s)
    print("eignevlaues for symmetric")
    print(eig_s)
    print("Period symmetric")
    print(P)
    print("Half period symmetric")
    print(T12)
    print("omega_0 for symmetric")
    print(omega_0)
    print("dampening ration for symmetric")
    print(damp)

    plt.figure("TAS")
    plt.plot(T,u)
    plt.plot(T,tas)
    plt.xlabel("time[s]")
    plt.ylabel("TAS [m/s]")

    plt.figure("AoA")
    plt.plot(T,alpha)
    plt.plot(T,aoa)
    plt.xlabel("T[s]")
    plt.ylabel("AoA [rad]")

    plt.figure("flight path angle")
    plt.plot(T,theta)
    plt.plot(T,pitch)
    plt.xlabel("T[s]")
    plt.ylabel("flight path angle [rad]")

    plt.figure("pitchrate ")
    plt.plot(T,q)
    plt.plot(T,pitch_rate)
    plt.xlabel("T[s]")
    plt.ylabel("pitchrate [rad/s]")

   
    plt.show()

if a_sym:
    
    data,T = fd.get_data_eigen(2)

    aoa = np.transpose(np.array(data[0]))[0]*np.pi/180
    tas = np.transpose(np.array(data[1]))[0]
    pitch = np.transpose(np.array(data[3]))[0]*np.pi/180
    roll = np.transpose(np.array(data[2]))[0]*np.pi/180
    yaw = np.transpose(np.array(data[4]))[0]
    roll_rate = np.transpose(np.array(data[5]))[0]*np.pi/180
    yaw_rate = np.transpose(np.array(data[7]))[0]*np.pi/180
    uv = np.transpose([np.transpose(np.array(data[8]))[0]*np.pi/180,np.transpose(np.array(data[10]))[0]*np.pi/180])
    hp = np.transpose(np.array(data[11]))[0]*np.pi/180

    cs = cp.Cit_par(aoa[0],hp[0],tas[0],pitch[0])


    A_a,B_a,C_a,D_a = st.get_ss_assymetric(aoa[0],hp[0],tas[0],pitch[0])
    sys_a = ct.lti(A_a,B_a,C_a,D_a)
    
    #plt.plot(aoa)
    plt.plot(np.transpose(uv)[1])
    plt.show()

    t1, y1, x1  = ct.lsim(sys_a,uv,T,[yaw[0],roll[0],yaw_rate[0],roll_rate[0]])

    beta = y1[:,0]
    phi = y1[:,1]
    p = y1[:,2]*cs.V0/(2*cs.b)
    r = y1[:,3]*cs.V0/(2*cs.b)

    eig_a = np.linalg.eig(A_a)[0]
    T12 = -0.693/(np.real(eig_a))*cs.c/cs.V0
    omega_0 = np.sqrt(np.real(eig_a)**2+np.imag(eig_a)**2)*cs.V0/cs.c
    P = 2*np.pi/np.imag(eig_a)*cs.c/cs.V0
    damp = -np.real(eig_a)/np.sqrt(np.real(eig_a)**2+np.imag(eig_a)**2)

    print("A matrix for asymmetric:")
    print(A_a)
    print("B matrix for asymmetric:")
    print(B_a)
    print("eignevlaues for asymmetric")
    print(eig_a)
    print("Period asymmetric")
    print(P)
    print("Half period asymmetric")
    print(T12)
    print("omega_0 for asymmetric")
    print(omega_0)
    print("dampening ration for asymmetric")
    print(damp)

    plt.figure("Yaw angle/rudder")
    plt.plot(T,beta)
    plt.plot(T,yaw)
    plt.xlabel("time[s]")
    plt.ylabel("Yaw [rad]")

    plt.figure("roll angle/rudder")
    plt.plot(T,phi)
    plt.plot(T,roll)
    plt.xlabel("T[s]")
    plt.ylabel("roll [rad]")

    plt.figure("Yaw rate/rudder")
    plt.plot(T,p)
    plt.plot(T,yaw_rate)
    plt.xlabel("T[s]")
    plt.ylabel("yaw rate [rad/s]")

    plt.figure("roll rate/rudder ")
    plt.plot(T,r)
    plt.plot(T,roll_rate)
    plt.xlabel("T[s]")
    plt.ylabel("roll rate [rad/s]")

    plt.show()
