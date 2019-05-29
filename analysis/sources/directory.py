import glob
def filepath_source(path,start=0):
    """ Wrapper around glob.glob function """
    return (f for f in glob.glob(path)[start:])

def read_image(filepaths):
    for path in filepaths:
        with Image.open(path) as im:
            yield im

