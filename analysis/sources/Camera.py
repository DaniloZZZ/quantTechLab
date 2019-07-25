from queue import LifoQueue as Queue
import queue, time
from multiprocessing import dummy as thr

def Camera_source(cam):
    """
    An iterator over images from cv.VideoCapture
    """
    n = 0
    while True:
        if cam.isOpened():
            try:
                ret, frame = cam.read()
            except Exception as e:
                print(str(e))
                time.sleep(0.1)
            if ret:
                yield frame
            else:
                print("Nothing")
        else:
            print("Camera is closed! Stopping")
            cam.release()
            return

def queue_put(gen, q, **kw):
    for i in gen:
        q.put(i,**kw)

def queue_source(q, **kw):
    while True:
        try:
            yield q.get(**kw)
        except queue.Empty:
            yield None

def thr_camera(cam, q_size=1, block_get=False):
    q = Queue(maxsize=q_size)
    def loop():
        c = Camera_source(cam)
        queue_put(c, q)
    p = thr.Process(
            target=loop,
            args=()
            )
    p.start()
    source = queue_source(q, block=block_get)
    return source, p
