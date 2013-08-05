import random
import bisect

def bin_search(l,v):
    values, categories = zip(*l)
    return categories[bisect.bisect(values,v)]

def pdf_to_cdf(categorial):
    cdf = []
    total = 0.0
    for (p, x) in categorial:
        total += p
        cdf.append((total,x))
    return cdf

def sample(categorial, n=1):
    """
    Pick a random sample from the categorial distribution.
    """
    cdf = pdf_to_cdf(categorial)
    result = [] 
    for i in range(n):
        t = random.random()
        result.append(bin_search(cdf, t))
    return result[0] if n==1 else result
