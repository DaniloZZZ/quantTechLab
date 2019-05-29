import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from sklearn.decomposition import PCA as PCA_sk
import numpy as np

def pca_sklearn(data,n_components=10):
    """
    Perform PCA decomosition
    Parameters
    ----------
    data: iterable
        data to get images from, numpy array
    n_components: int
        number of remaining components

    Returns
    --------
    (sklearn.PCA, np.array)
        PCA fitted transformer and transformed data
    """
    pca = PCA_sk(n_components=n_components, random_state=42)
    X = [d.flatten() for d in data]
    transformed = pca.fit_transform(X)
    return pca,transformed

def get_pca_sk_transformer(data,n_components=10):
    """
    Get Generator that transforms pca
    Parameters
    ----------
    data: iterable
        data to get images from, numpy array
    n_components: int
        number of remaining components

    Returns
    --------
    (generator, np.array)
        generator of transformed images and transformed train data
    """
    pca,transformed = pca_sklearn(data,n_components=n_components)
    def generator(data):
        for image in data:
            x = image.flatten().reshape(1,-1)
            yield pca.transform(x)
    return generator, transformed

def splitter(data,parts=6):
    """
    a generator of image parts, every image gets splitted
    to `parts`**2 parts. 
    Parameters
    ----------
    data: iterable
        data to get images from
    parts: int
        specify ratio of inital image and small

    yields
    --------
    np.array
        parts of image
    """
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
    X=np.array([l-np.mean(l) for l in X]) 
    U,s,V=np.linalg.svd(X)                
    eps=np.sort(s)[-num_components]       
    E = np.array([vec for val,vec in zip(s,V)
                  if val>eps])            
    X_=np.dot(E,X.T).T                    
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
