import Cit_par as cs
import numpy as np

def get_ss_symmetric():
    # variable_vector [u,alpha,theta,q,de]
    x_a = np.array([cs.CXu, cs.CXa, cs.CZ0, cs.CXq, cs.CXde])
    x = cs.V0/(2*cs.c*cs.muc)*x_a
    z_a = np.array([cs.CZu, cs.CZa, -cs.CX0, 2*cs.muc+cs.CZq, cs.CZde])
    z = cs.V0/(cs.c*(2*cs.muc-cs.CZadot))*z_a
    k = cs.Cmadot/(2*cs.muc-cs.CZadot)
    m_a =np.array([cs.Cmu + cs.CZu * k, cs.Cma + cs.CZa * k, -cs.CX0*k, cs.Cmq + cs.CZadot*k, cs.Cmde + cs.CZde*k])
    m = cs.V0/(2*cs.c*cs.muc*cs.KY2)*m_a

    A_s = np.array([[x[0],x[1],x[2],  0],
                    [z[0],z[1],z[2],z[3]],
                    [   0,   0,   0,cs.V0/cs.c],
                    [m[0],m[1],m[2],m[3]]])

    B_s = np.array([[x[4]],[z[4]],[0],[m[4]]])
    return A_s, B_s

def get_ss_assymetric():
    # variable_vector [beta,phi,p,r,da,dr]
    y_a = np.array([cs.CYb, cs.CL, cs.CYp, cs.CYr-4*cs.mub, cs.CYda, cs.CYdr])
    y = cs.V0/(2*cs.b*cs.mub)*y_a
    l_a = np.array([[cs.Clb, 0, cs.Clp, cs.Clr, cs.Clda, cs.Cldr],
                    [cs.Cnb, 0, cs.Cnp, cs.Cnr, cs.Cnda, cs.Cndr]])
    l = cs.V0/(4*cs.b*cs.mub)*1/(cs.KX2*cs.KZ2-cs.KXZ)*(l_a[0]*cs.KZ2+l_a[1]*cs.KXZ)
    n = cs.V0/(4*cs.b*cs.mub)*1/(cs.KX2*cs.KZ2-cs.KXZ)*(l_a[1]*cs.KZ2+l_a[0]*cs.KXZ)

    
    A_a = np.array([[ y[0],y[1],y[2],y[3]],
                    [ 0, 0,2*cs.V0/cs.c,0],
                    [ l[0],   0,l[2],l[3]],
                    [ n[0],   0,n[2],n[3]]])
    B_a = np.array([[  0,y[5]],
                    [  0,   0],
                    [l[4],l[5]],
                    [n[4],n[5]]])
    return A_a, B_a