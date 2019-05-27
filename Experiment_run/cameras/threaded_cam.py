import multiprocessing.dummy as thr
#[[import multiprocessing as thr
import cv2
import time
import queue

def camPreview(previewName,cam):
    cv2.namedWindow(previewName)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

def read_once(cam):
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    if rval:
        return frame

class Camera:
    def __init__(self, number, name=None, buf_size=3):
        if not name:
            self.name = "Camera"+str(number)
        else:
            self.name = name
        self.number = number
        self.buffer = queue.Queue(buf_size)

    def open(self):
        self.cam = cv2.VideoCapture(self.number)
        if not self.cam.isOpened():
            print("Failed to init")
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        # 10 was found to be the smallest value
        self.cam.set(cv2.CAP_PROP_EXPOSURE, 10)
        print("Param::auto exp?",self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        print("Param::exp time", self.cam.get(cv2.CAP_PROP_EXPOSURE))

    def close(self):
        if self.cam:
            self.cam.release()
            self.cam = None
        else:
            print("calling camera on already closed")

    def listen_open(self):
        while True:
            print("trying to open", self.name)
            self.open()
            if self.cam.isOpened():
                print("Opened",self.name)
                return
            time.sleep(1)

    def loop(self):
        self.listen_open()
        while True:
            frame = read_once(self.cam)
            if frame is None:
                self.listen_open()
            else:
                self.buffer.put(frame)
        self.close()

    def start(self):
        p = thr.Process( target =self.loop, args = ())
        p.start()
        return p

def start_cameras(idxs):
    cams = [Camera(i) for i in idxs]
    procs = [ c.start() for c in cams]
    for p in procs:
        p.join()

def get_cameras():
    arr = []
    for index in range(10):
        print(index)
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            continue
        else:
            arr.append(index)
        cap.release()
    return arr

def main():
    camsid = get_cameras()
    print(camsid)
    start_cameras(camsid)

if __name__=="__main__":
    main()
