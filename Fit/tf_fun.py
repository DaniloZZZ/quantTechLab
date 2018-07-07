import tensorflow as tf
import numpy as np

def init():
    expi = lambda x: tf.exp(tf.complex(0.,x))
    """
    def expi(x):
        if isinstance(x,tf.Variable) or isinstance(x,tf.Tensor):
            return tf.exp(tf.complex(0.,tf.cast(x,tf.float32)))
        else:
            return tf.exp(tf.complex(0.,float(x)))
    """
    #expi = lambda x: np.exp(1j*x)
    real = lambda x: tf.complex(x,0.)

    T = real(0.6)
    t = real(tf.sqrt(0.2))
    r = real(tf.sqrt(1.-0.2))
    a = (33.)
    b =(39.)
    c =(39.)

    def a01(al=np.pi, thet=0):
        return expi(thet)*\
                tf.Variable(
                    [
                        [t, r*expi(al), 0.],
                        [-r*expi(-al), t, 0.],
                        [0., 0., expi(-thet)]
                    ]
                )

    def a12(al= np.pi, thet=0):
        return expi(thet)*\
                tf.Variable(
                    [
                        [ expi(-thet), 0, 0],
                        [0, t, r*expi(al)  ],
                        [0, -r*expi(-al), t]
                    ]
                )

    def ph(a,b,c):
        #l = [float(x) if isinstance(x,int) else x for x in[a,b,c] ]
        l = [a,b,c]
        return tf.diag([expi(x) for x in l])
    def ph2(phi):
        return ph(0.,phi,2.*phi)
    def refl0(k):
        return tf.diag([real(tf.sqrt(1-k)),real(1.),real(1.)])
    def refl1(k):
        return tf.diag([real(1.),real(tf.sqrt(1-k)),real(1.)])
    def refl2(k):
        return tf.diag([real(1.),real(1.),real(tf.sqrt(1-k))])

    def prob_part(*x,phi=0.1,alp = 0.125):
        pi = np.pi
        vals = [0.125,0,pi,0,pi,pi,pi,0,0,0,0,0]
        vals = [tf.constant(float(x),dtype=tf.float32) for x in vals]
        a1,al1,al2,al3,al4,ala,alb,th1,th2,th3,th4,thb = vals
        psi=1.33
        op = tf.diag([real(1.),real(1.),real(1.)])
        matr = [
            a12(al4,th4),
            ph(0.,np.arctan(1.41)-2*psi+al4-th3-th2-th1-al1+al2+alb+ x[3],0.),
            refl1(a1),
            ph(0.,psi,psi),
            a01(al3,th3),
        ]
        i=0
        for m in matr:
            i+=1
            op = tf.matmul(op,m)

        phi = tf.cast(phi,tf.float32)

        mod = tf.Variable([[a],[b],[c]],dtype=tf.complex64)
        mod = tf.divide(mod,tf.norm(mod))
        ret = tf.matmul(op,ph2(phi))
        ret = tf.matmul(ret,mod)

        return ret
    return prob_part

