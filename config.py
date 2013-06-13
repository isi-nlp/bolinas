class Config(dict):

    def __init__(cls, *args, **kwargs):
        pass
    
    def load_config(self, f):
        """
        Update this Config object by reading configuration objects from file f.
        """
        #TODO
        pass 

config = Config()
config.maxk = 1000

