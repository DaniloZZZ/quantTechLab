
import numpy as np
import matplotlib.pyplot as plt

def LossLandscape(function, points, dots = 30,scale=1,figsize=(10,10)):
    """
    Plots a surface sliced by a plane tthat intercepts 3 points
    """
    p0, p1, p2 = np.array(points).reshape(3,-1,1)

    assert len(p0)==len(p1)==len(p2)
    t = np.linspace(-scale/2, scale/2 ,dots)
    mesh = np.array(np.meshgrid(t,t)).transpose(2,1,0)
    grid = mesh.dot(np.array([p1,p2]).reshape(2,-1))+p0.reshape(12)
    grid = grid.reshape(dots,dots,-1)
    #grid = grid.reshape(dots,dots,-1)
    #print('xy',x.shape,y.shape)
    print('grid shape', grid.shape)
    values = np.array([[function(p) for p in line] for line in grid])
    values = values.reshape(dots,dots)
    print('val shape',values.shape)

    plt.figure(figsize=figsize)
    t = t.reshape(-1)
    plt.contourf(t,t,values,40, cmap='inferno')
    return grid, values

