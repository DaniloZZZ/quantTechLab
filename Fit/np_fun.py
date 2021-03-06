import numpy as np

T = 0.445
t = np.sqrt(T)
r = np.sqrt(1-T)

a = np.sqrt(T*(1-T))
b = (1-T)
c = np.sqrt(T)
#a = b = c =1

def a01(al=np.pi, thet=0):
    return np.exp(1j*thet)*\
    np.array(
	[
	[t, r*np.exp(1j*al), 0],
	[-r*np.exp(-1j*al), t, 0],
	[0, 0, np.exp(-1j*thet)]
	]
    )

def a12(al=np.pi, thet=0):
    return np.exp(1j*thet)*\
    np.array(
	[
	[np.exp(-1j*thet), 0, 0],
	[0, t, r*np.exp(1j*al)],
	[0, -r*np.exp(-1j*al), t]
	]
    )

def a02(al=np.pi, thet=0):
    return np.exp(1j*thet)*\
    np.array(
	[
	[t, 0, r*np.exp(1j*al)],
	[0, np.exp(-1j*thet), 0],
	[-r*np.exp(-1j*al), 0, t]
	]
    )

def ph(a,b,c):
    return np.diag(np.exp(1j*np.array([a,b,c])))

def ph2(phi):
    if not isinstance(phi,np.ndarray):
        return ph(0,phi,2*phi)
    else:
        return np.array(
            [ph(0,i,2*i) for i in phi]
        )
def refl0(k):
    return np.diag([np.sqrt(1-k),1,1])
def refl1(k):
    return np.diag([1,np.sqrt(1-k),1])
def refl2(k):
    return np.diag([1,1,np.sqrt(1-k)])

def device_function(x,phi):
    return np.abs(np.square(prob_part(*x,phi=phi)))

def prob_part(*x,phi=0.1,alp = 0.125):
    a1 = 0.125
    asq2 = np.arctan(np.sqrt(2))
    g1,g2,g3 = 0.15, 0.2, 0.28
    op = np.diag([1,1,1])
    matr = [
        #np.array([0,0,1]),
        a12(np.pi/2, 0),
        ph(0., np.pi/2 + asq2 + x[3], 0),
        refl1(a1),

        a01(np.pi/2, 0),
        ph(np.pi - 2*asq2 + x[2], 0, 0),
        refl0(a1),

        a01(np.pi/2, 0),
        ph(0, x[1], 0),
        refl1(a1),

        a12(np.pi/2, 0),
        ph(0, 0, np.pi*3/2 + x[0]),
        refl2(a1),
        ph2(phi),
        refl2(g3),
        refl2(g2),
        refl1(g1),

    ]
    for m in matr:
        op = np.matmul(op,m)

    mod = np.array([a,b,c])
    mod = mod/np.linalg.norm(mod)
    #ret = np.matmul(op,ph2(phi))
    ret = np.matmul(op,mod)

    #ret=ret.reshape(ret.shape[-1],-1)
    return ret

def prob(*x,phi=0.1,alp = 0.125):
    a1,al1,al2,al3,al4,ala,alb,th1,th2,th3,th4,thb = 0.125,0,np.pi,0,np.pi,np.pi,np.pi,0,0,0,0,0
    psi=12
    op = a12(al4,th4)\
.dot(ph(0,np.arctan(1.41)-2*psi+al4-th3-th2-th1-al1+al2+alb+ x[3],0))\
.dot(refl1(a1))\
.dot(ph(0,psi,psi))\
.dot(a01(al3,th3))\
\
.dot(ph(-2*np.arctan(1.41)+al3-al2+np.pi+x[2],0,0))\
.dot(refl0(a1))\
.dot(ph(psi,psi,0))\
.dot(a01(al2,th2))\
\
.dot(ph(0,-al2+np.pi/2.-th1-alb+x[1],0))\
.dot(refl1(a1))\
.dot(ph(ala+thb+psi,psi,0))\
.dot(a12(al1,th1))\
\
.dot(ph(0,0,-al1-psi+ala+thb+alb+x[0]))\
.dot(refl2(a1))\
.dot(ph(0,ala+thb+alb,psi))

    mod = np.array([a,b,c])
    mod = mod/np.linalg.norm(mod)
    ret = op.dot(ph2(phi)).dot(mod)
    return ret
