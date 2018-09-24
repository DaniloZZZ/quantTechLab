import numpy as np
from PIL import Image
import re, time, os

def dgen(path,exp,start=0,limit = 10):
    skipped = 0
    for f in list(os.listdir(path))[start:start+limit]:
        image = Image.open(os.path.join(path,f))
        val = get_value(f,exp)
        if val:
            yield (image,val)
        else:
            skipped+=1
    print("Returned %i, skipped %i files"%(limit-skipped,skipped))
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
    t = time.mktime(tuple(date_props+[0,0,0]))
    #print(_unix2str(t))
    return t

def _unix2str(t):
    return time.asctime(time.localtime(t))
