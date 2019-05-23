#import multiprocessing.dummy as thr
import multiprocessing as thr
import cv2


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
    else:
        rval = False
    if rval:
        return frame

class Camera:
    def __init__(self, number, name=None):
        if not name:
            self.name = "Camera"+str(number)
        else:
            self.name = name
        self.number = number

    def open(self):
        self.cam = cv2.VideoCapture(self.number)

    def close(self):
        if self.cam:
            self.cam.release()
            self.cam = None
        else:
            print("calling camera on already closed")

    def loop(self):
        self.open()
        frame = read_once(self.cam)
        print(frame)
        #camPreview(self.name,self.cam)
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

def main():
    start_cameras([0,2])

if __name__=="__main__":
    main()
