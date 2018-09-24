import numpy as np
from PIL import Image
import re, time, os, cv2
from tqdm import tqdm

def dgen(path,exp,start=0,limit = 10):
    skipped,ret = 0,0
    for f in list(os.listdir(path))[start:start+limit]:
        val = get_value(f,exp)
        if val:
            image = Image.open(os.path.join(path,f))
            ret+=1
            yield (image,val)
        else:
            skipped+=1
    print("Returned %i, skipped %i files"%(ret,skipped))

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
    i,v_0= data[0]
    pack = []
    bucket = []
    b_sizes = []
    for i,v in data:
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
    if sum(b_sizes)!=len(data):
        print("error",sum(b_sizes),len(data))
    return pack


def get_value(f,exp_data):
    st = exp_data.start_time
    dt = exp_data.tick
    tr = exp_data.transition_dur
    t = _parse_time(f)
    if not t:
        return None
    val = t//dt
    since_last_tick = t%dt
    if since_last_tick<=tr:
        return None
    else: return val

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
