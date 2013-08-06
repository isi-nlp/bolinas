import math

# Compute sum of logs, courtesy of David Chiang

LOGZERO=-1e100

def logadd(lp, lq):
    if lp > lq:
        return lp + math.log1p(math.exp(lq - lp))
    else:
        return lq + math.log1p(math.exp(lp - lq))

def logsum(iterable):
    lp = LOGZERO
    for lq in iterable:
        lp = logadd(lp, lq)
    return lp
