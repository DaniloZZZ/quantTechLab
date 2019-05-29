import numpy as np
from PIL import Image
import re, time, os, cv2, time
from tqdm import tqdm

def data_generator(source,exp):
    """
    A generator of images data
    Parameters
    ----------
    source: generator
        A generator which yields image and name (Pillow.Image,string)
    exp: Object
        Experiment data

    Yields
    --------
    (Pillow.Image, int)
        Image and it's target value
    """
    skipped,ret = 0,0
    for image,name in source:
        val = get_value(name,exp)
        #print('file:',name,'value:',val)
        if val:
            ret+=1
            yield (np.array(image),val)
        else:
            skipped+=1
    print("DataGen returned %i, skipped %i files"%(ret,skipped))

def resize_tuple(data,factor=0.3):
    datagen = (i for i in data)
    for  d in datagen:
        yield (resize(d[0],factor), d[1:])

def resize(imdata,factor=0.3):
    image = imdata
    height, width = image.shape[:2]
    resized_image=cv2.resize(image,(int(factor*width),int(factor*height)))
    return resized_image

def downscale_nsave(path,d,factor=0.3):
    out = os.path.join(path,d)
    if not os.path.exists(out): 
        os.makedirs(out) 
    for f in tqdm(list(os.listdir(path))):
        f_ = os.path.join(path,f)#.replace('\\','/')
        #print(f_)
        image = cv2.imread(f_)
        height, width = image.shape[:2]
        resized_image=cv2.resize(image,(int(factor*width),int(factor*height)))
        cv2.imwrite(os.path.join(out,f),resized_image)



def pack_by_value(data):
    # convert this to generator always
    data = (x for x in data)
    i,v_0= next(data)
    data_len = 0
    pack = []
    bucket = []
    b_sizes = []
    for i,v in data:
        data_len+=1
        if v==v_0:
            bucket.append(i)
        else:
            pack.append((v_0,bucket))
            v_0 = v
            b_sizes.append(len(bucket))
            bucket = [i]
    if len(bucket)!=0:
        pack.append((v_0,bucket))
        b_sizes.append(len(bucket))

    print("Packed to %i buckets, lengths: %s"%(len(b_sizes),str(b_sizes)))
    if sum(b_sizes)!=data_len:
        print("error",sum(b_sizes),data_len)
    return pack


def get_value(f,exp_data):
    st = exp_data.start_time
    dt = exp_data.tick
    tr = exp_data.transition_dur
    t = _parse_time(f)
    if not t:
        return None
    val = t//dt
    start = st//dt
    since_last_tick = t%dt
    if since_last_tick<=tr:
        return None
    if val<start:
        return None
    else: return val-start

def _parse_time(f):
    regexp = "([0-9]{2,4})(-|\n|\()?"
    m = re.findall(regexp,f)
    date_props = []
    for num,_ in m:
        date_props.append(int(num))
    if len(date_props)!=6:
        return None
    t = time.mktime(tuple(date_props+[0,0,0]))
    #print(_unix2str(t))
    return t

def _unix2str(t):
    return time.asctime(time.localtime(t))
