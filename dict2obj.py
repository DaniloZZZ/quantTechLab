########################################################################
class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """
 
    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])
 
    #----------------------------------------------------------------------
    def __repr__(self):
        """"""
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s>" % attrs
 
#----------------------------------------------------------------------
if __name__ == "__main__":
    ball_dict = {"color":"blue",
                 "size":"8 inches",
                 "material":"rubber"}
    ball = Dict2Obj(ball_dict)########################################################################
class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """
 
    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])
 
    #----------------------------------------------------------------------
    def __repr__(self):
        """"""
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s>" % attrs
 
#----------------------------------------------------------------------
if __name__ == "__main__":
    ball_dict = {"color":"blue",
                 "size":"8 inches",
                 "material":"rubber"}
    ball = Dict2Obj(ball_dict)
