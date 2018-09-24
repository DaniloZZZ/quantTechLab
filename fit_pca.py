import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_images_table(imgs,figsize=(10,10), columns=8):
    n=len(imgs)
    h = (n-1)//columns+1
    fig = plt.figure(1,figsize)

    grid=ImageGrid(fig,111,
            (h,columns),
            axes_pad=0.05
            )
    for i in range(n):
        grid[i].imshow(imgs[i],cmap='gray')
    plt.show()

def ___plot_images_table(imgs,figsize=(10,10), columns=8):
    n=len(imgs)
    h = (n-1)//columns+1
    f,ax=plt.subplots(h,columns,figsize=(figsize[0],figsize[1]*h/columns))
    for i in range(1,n+1):
        ax[(i-1)//columns,i%columns-1].imshow(imgs[i-1],cmap='gray')
    plt.show()
