import random
import bisect
import math
from common.logarithm import LOGZERO, logadd

def bin_search(l,v):
    values, categories = zip(*l)
    index = bisect.bisect(values,v)    
    return categories[index]

def pdf_to_cdf(categorial):
    cdf = []
    total = None 
    for (p, x) in categorial:
        if total is None: 
            total = p
        else: 
            total = logadd(total,p)
        cdf.append((total,x))
    return cdf

def sample(categorial, n=1):
    """
    Pick a random sample from the categorial distribution.
    """
    cdf = pdf_to_cdf(categorial)
    result = [] 
    for i in range(n):
        r = random.random()
        if r == 0:
            t = LOGZERO
        else: 
            t = math.log(r)
        result.append(bin_search(cdf, t))
    return result[0] if n==1 else result
