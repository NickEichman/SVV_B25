import Cit_par as cs
import numpy as np

def get_ss_symmetric():
    # variable_vector [u,alpha,theta,q,de]
    # (forward velocity,angle of attack, pitchangle, ptichrate, elevator deflection)
    #implementation of the statespace matrices as defined in Lecture notes AE3202 March 2013
    x_a = np.array([cs.CXu, cs.CXa, cs.CZ0, cs.CXq, cs.CXde])
    x = cs.V0/(2*cs.c*cs.muc)*x_a
    z_a = np.array([cs.CZu, cs.CZa, -cs.CX0, 2*cs.muc+cs.CZq, cs.CZde])
    z = cs.V0/(cs.c*(2*cs.muc-cs.CZadot))*z_a
    k = cs.Cmadot/(2*cs.muc-cs.CZadot)
    m_a =np.array([cs.Cmu + cs.CZu * k, cs.Cma + cs.CZa * k, -cs.CX0*k, cs.Cmq + (2*cs.muc+cs.CZq)*k, cs.Cmde + cs.CZde*k])
    m = cs.V0/(2*cs.c*cs.muc*cs.KY2)*m_a

    A_s = np.array([[x[0],x[1],x[2],  0],
                    [z[0],z[1],z[2], z[3]],
                    [   0,   0,   0,(cs.V0/cs.c)],
                    [m[0],m[1],m[2], m[3]]])

    B_s = np.array([[x[4]],[z[4]],[0],[m[4]]])
    C_s = np.eye(4)
    D_s = np.zeros([4,1])
    return A_s, B_s, C_s, D_s

def get_ss_assymetric():
    # variable_vector [beta,phi,p,r,da,dr]
    # (yaw angle,roll angle,rollrate, yawrate, aileron deflection,rudder deflection)
    #implementation of the statespace matrices as defined in Lecture notes AE3202 March 2013
    y_a = np.array([cs.CYb, cs.CL, cs.CYp, cs.CYr-4*cs.mub, cs.CYda, cs.CYdr])
    y = cs.V0/(2*cs.b*cs.mub)*y_a
    l_a = np.array([cs.Clb, 0, cs.Clp, cs.Clr, cs.Clda, cs.Cldr])
    n_a = np.array([cs.Cnb, 0, cs.Cnp, cs.Cnr, cs.Cnda, cs.Cndr])
    l = cs.V0/(4*cs.b*cs.mub)*1/(cs.KX2*cs.KZ2-cs.KXZ**2)*(l_a*cs.KZ2+n_a*cs.KXZ)
    n = cs.V0/(4*cs.b*cs.mub)*1/(cs.KX2*cs.KZ2-cs.KXZ**2)*(n_a*cs.KX2+l_a*cs.KXZ)

    
    A_a = np.array([[ y[0],y[1],y[2],y[3]],
                    [ 0, 0,2*cs.V0/cs.b,0],
                    [ l[0],   0,l[2],l[3]],
                    [ n[0],   0,n[2],n[3]]])
    B_a = np.array([[  0,y[5]],
                    [  0,   0],
                    [l[4],l[5]],
                    [n[4],n[5]]])
    C_a = np.eye(4)
    D_a = np.zeros([4,2])
    return A_a, B_a, C_a, D_a