
def Levenh_source(path):
    """
    a generator of files from directory where levenhuk writes.
    checks the path for updates in eternal loop, then yields any
    updates
    parameters
    ----------
    path: string
        a path where to look for new files

    yields
    --------
    (pillow.image, string)
        image and it's name
    """
    give_up_time = 7.0
    files = []
    time_upd = time.time()
    started = False
    num = 0
    print("Watching %s for new files..."%path)
    while True:
        for f in os.listdir(path):
            if f not in files:
                files.append(f)
                num+=1
                image = None
                time.sleep(0.1)
                try:
                    with Image.open(os.path.join(path,f)) as im:
                        image= im
                        time_upd = time.time()
                        if not started:
                            print("found first image, running ok...")
                        started = True
                        yield (image,f)
                except PermissionError:
                    time_upd = time.time()
                    print("ERROR: file %s permission denied"%f)
                    pass
                except FileNotFoundError:
                    print("ERROR: file %s not found"%f)
                    pass
                except IOError:
                    print("ERROR: file %s IOerrr"%f)
                    pass
                #print('yielding image',f)
            else:
                since_update =time.time()-time_upd 
                if since_update>give_up_time:
                    print("update was %d seconds ago, giving up"%since_update)
                    return -1
                pass
