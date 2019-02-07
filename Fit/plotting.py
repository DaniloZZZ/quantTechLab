
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

sampl = lambda f,fm=0,to=10,cnt=100: [np.linspace(fm,to,cnt),list(map(f,np.linspace(fm,to,cnt)))]

def set_fonts_ticks(fig,ax):
    fontsize =14
    try:
        ax.set_xlabel(' φ ',fontsize=fontsize)
        #ax.set(ylabel=' Intensity ')
    except:
        plt.ylabel(' Intensity ', fontsize=fontsize)
        plt.xlabel(' φ ',fontsize=fontsize)

    tick_width =1
    ax.tick_params(axis="y",direction="in",labelsize=13,width=tick_width,length=8)
    ax.tick_params(axis="x",direction="in",labelsize=13,width=tick_width, length=8)

def plot3(func=None, data=None, on_same=False):
    if func:
        if data:
            domain_used = data[0][1]
        else:
            domain_used = np.linspace(0,12,200)
        domain, func_vals = sampl(func,domain_used[0],domain_used[-1],200)
        func_vals = np.array(func_vals).T

    figsize=(16,4)
    if on_same:
        plt.figure(figsize=figsize)
        ax =[plt,plt,plt]
        f = plt
    else:
        f,ax = plt.subplots(1,3,figsize=figsize,sharey=True)
    colors  = ['blue','red','green']
    colors  = [cm.brg(i) for i in np.linspace(0,1,3)]

    for i,color in zip(range(3),colors):
        if data:
            y, x = data[i]
            errx = 0.02+0.06*y
            ax[i].fill_between(x,y+errx,y-errx,
                    color='green',alpha=0.1,label='error region')
            ax[i].errorbar(x, y, xerr=4*np.pi/50, yerr=errx,
                    capthick=0.2,elinewidth=0.5,fmt='none',
                    color='gray',capsize=1.5)
            set_fonts_ticks(f,ax[i])
            ax[i].plot(x, y,'.',markersize=1,color='black',
                    label=r'data $|%i\rangle$'%i)
        if func:
            ax[i].plot(domain,func_vals[i],
                    label=r'function $|%i\rangle$'%i,color=color)
    return ax

