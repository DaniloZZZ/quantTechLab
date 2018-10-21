import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import numpy as np

def splitter(data,parts=6):
    for img in data:
        w,h = img.width,img.height
        dw,dh = w//parts, h//parts
        data= np.array(img)
        for idx in range(parts**2):
            x,y = idx%parts,idx//parts
            yield data[y*dh:(y+1)*dh,x*dw:(x+1)*dw]

def merger(patches,parts=6):
    idx = 0
    image=None
    for p in patches:
        if idx>=parts**2:
            idx = 0
            yield image
        if idx==0:
            image = np.zeros((p.shape[0]*parts,p.shape[1]*parts))
            dh,dw = p.shape
        x,y = idx%parts,idx//parts
        image[y*dh:(y+1)*dh,x*dw:(x+1)*dw]=p
        idx+=1

def pca_core(X,num_components):
    X=np.array([l-np.mean(l) for l in X]) # Вычитаем среднее
    U,s,V=np.linalg.svd(X)                # Находим собственные вектора
    eps=np.sort(s)[-num_components]       # пороговое собственное значение (служебная строчка) 
    E = np.array([vec for val,vec in zip(s,V)
                  if val>eps])            # берем только важные вектора   (с большими собственными значениями)
    X_=np.dot(E,X.T).T                    # Преобразуем данные
    return X_,E,s

def get_PCA_gen(dset,factor=0.5,parts=6,num_components=10):
	patches= list(splitter( resizer(dset, factor=factor), parts=parts ))
	X = np.array([i[:,:,1].flatten() for i in patches])
	transformed, operator, s = pca_core(X, num_components)
	def generator(data):
		for image in data:
			yield np.dot(E,image)
	return generator, transformed

def resizer(images,factor=0.5):
    for i in images:
        w,h = i.width,i.height
        yield i.resize((int(w*factor),int(h*factor)), Image.ANTIALIAS)

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
