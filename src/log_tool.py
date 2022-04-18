
import time


def log_td_filename():
    return 'log-mini-'+time.strftime("%Y%m%d-%H%M%S")+'.txt'

def log_name_filename(base_name):
    return 'log-mini-'+base_name+'.txt'

def log_name_td_filename(base_name):
    return 'log-mini-'+base_name+'-'+time.strftime("%Y%m%d-%H%M%S")+'.txt'


"""
def log(f):
    def func_wrapper(*args, **kwargs):
        self = args[0]
        print 'f:', f.__name__
        if self.log:
            self.log.write('["{}", "{}",{},{}]'.format(self.log_prefix, f.__name__,args[1:],kwargs)+'\n')
        return f(*args, **kwargs)
    return func_wrapper
"""

from functools import wraps

def log(f):
    @wraps(f)
    def func_wrapper(*args, **kwargs):
        self = args[0]
        self.log.write('["{}",{},{}]'.format(f.__name__,args[1:],kwargs)+'\n')
        self.log.flush()
        def u():
            f(*args, **kwargs)
        self.log_stack.append(u)
        self.undo_stack.append(False)
        return f(*args, **kwargs)
    return func_wrapper


def undo_log(f):
    @wraps(f)
    def func_wrapper(*args, **kwargs):
        self = args[0]
        self.log.write('["{}",{},{}]'.format(f.__name__,args[1:],kwargs)+'\n')
        self.log.flush()
        undo, v = f(*args, **kwargs)
        def u():
            f(*args, **kwargs)
        self.log_stack.append(u)
        self.undo_stack.append(undo)
        return v
    return func_wrapper

