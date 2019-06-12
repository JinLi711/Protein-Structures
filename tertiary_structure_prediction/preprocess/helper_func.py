import time
from functools import wraps
import logging

logging.basicConfig(
    filename="preprocess.log",  
    level=logging.INFO,
    format='%(levelname)s - %(filename)s \n\t %(message)s')

def timing_val(func):
    @wraps(func)
    def wrapper(*arg, **kw):
        t1 = time.time()
        result = func(*arg, **kw)
        t2 = time.time()
        time_str = "TIME Function: {}: took: {}".format(
            func.__name__,
            (t2 - t1))
        return result, time_str
    return wrapper
