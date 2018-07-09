import numpy as np
from tqdm import tqdm

carr =lambda f,z: lambda x: f(z,x)
#sampl_map_depr = lambda f,fm=0,to=10,cnt=100: [np.linspace(fm,to,cnt),list(map(f,np.linspace(fm,to,cnt)))]
#sampl = lambda f,fm=0,to=10,cnt=100: [np.linspace(fm,to,cnt),f(np.linspace(fm,to,cnt))]
np_map = lambda f,x: np.array(list(map(f, x)))

class Annealer:
    def __init__(self,func,points_data,**params):
        self.__dict__=dict(params)

        self.func = func
        self.energy = params.get('energy',3)
        self.points,self.data =points_data
        self.cost = lambda f: np.sum(np.square(f(self.points)-self.data))
        #self.prob = lambda e,e_,t: 1/(1+np.exp(-((e-e_)*(1+5*np.heaviside(e_-e,0))/3-np.heaviside(e_-e,0)*0.01/t)))
        self.prob = lambda e,e_,t: 1/(1+np.exp(-((e-e_)*(1+0.1*np.heaviside(e_-e,0))/self.energy-0.3/t)))

    def opt(self,start_point,**par):
        """
        max_steps: maximum iteration count. :def:1000
        scales: a list of scales for every dimension. :def:np.ones(len(start_point))
        """
        dim = len(start_point)
        scales =par.get('scales',np.ones(dim))
        p = start_point
        temp = 10
        self.dots=[]
        self.costs,self.probs=[],[]
        self.best = p
        vmin=10000
        steps = par.get('max_steps',20000)
        for i in tqdm(range(steps)):
            temp = steps/(i+1)-0.9999
            n_p = np.random.randn(dim)
            p_ = n_p*scales*(1-0*i/5000) + p
            #count value of fun in two points
            f,f_ = carr(self.func,p),carr(self.func,p_)
            v,v_ = self.cost(f),self.cost(f_)

            #print('iter ',i,' np,p',p_,p,' cost:',v,v_)
            prob =self.prob(v,v_,temp)
            self.probs.append(prob)
            #print(prob)
            if(v_<vmin):
                vmin=v_
                self.best=p_
            if (prob>np.random.rand()):
                p = p_
                #print("MOVED")
                self.dots.append(p_)
                self.costs.append(v_)

        return p,self.cost(carr(self.func,p))

