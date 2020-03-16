#import control as ct
import numpy as np
import statespace as st
import matplotlib.pyplot as plt
import scipy.signal as ct
import Cit_par_ver as cs

sym =  True
a_sym = False

if sym:
    A_s,B_s,C_s,D_s = st.get_ss_symmetric()
    sys_s = ct.lti(A_s,B_s,C_s,D_s)
    T = np.linspace(0,10,101)

    uv= np.zeros(np.shape(T))
    for i in range(len(uv)):
        uv[i] = -0.005


    t1, y1, x1  = ct.lsim(sys_s,uv,T,[0,cs.alpha0,cs.th0,0])
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
    plt.xlabel("time[s]")
    plt.ylabel("TAS [m/s]")

    plt.figure("AoA")
    plt.plot(T,alpha)
    plt.xlabel("T[s]")
    plt.ylabel("AoA [rad]")

    plt.figure("flight path angle")
    plt.plot(T,theta)
    plt.xlabel("T[s]")
    plt.ylabel("flight path angle [rad]")

    plt.figure("pitchrate ")
    plt.plot(T,q)
    plt.xlabel("T[s]")
    plt.ylabel("pitchrate [rad/s]")

    plt.show()

if a_sym:
    A_a,B_a,C_a,D_a = st.get_ss_assymetric()
    sys_a = ct.lti(A_a,B_a,C_a,D_a)
    T = np.linspace(0,15,101)

    uv= np.zeros([np.shape(T)[0],2])
    for i in range(len(uv)):
        uv[i,1] = 0.025

    t1, y1, x1  = ct.lsim(sys_a,uv,T,[0,0,0,0])

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
    plt.xlabel("time[s]")
    plt.ylabel("Yaw [rad]")

    plt.figure("roll angle/rudder")
    plt.plot(T,phi)
    plt.xlabel("T[s]")
    plt.ylabel("roll [rad]")

    plt.figure("Yaw rate/rudder")
    plt.plot(T,p)
    plt.xlabel("T[s]")
    plt.ylabel("yaw rate [rad/s]")

    plt.figure("roll rate/rudder ")
    plt.plot(T,r)
    plt.xlabel("T[s]")
    plt.ylabel("roll rate [rad/s]")

    plt.show()
