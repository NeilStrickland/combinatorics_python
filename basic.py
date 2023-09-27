def interval_oo(a,b):
    """Elements of the interval (a,b) """
    return list(range(a+1,b))

def interval_oc(a,b):
    """Elements of the interval (a,b] """
    return list(range(a+1,b+1))

def interval_co(a,b):
    """Elements of the interval [a,b) """
    return list(range(a,b))

def interval_cc(a,b):
    """Elements of the interval [a,b] """
    return list(range(a,b+1))

def BB(n,size = None):
    """
    Binary sequences of length n.  If size is specified, only sequences with
    that number of ones are returned.
    """
    if size == None:
        B = [[]]
        for _ in range(n):
            B = [b+[x] for b in B for x in range(2)]
        return B
    else:
        if size == 0:
            return [[0 for _ in range(n)]]
        elif n == 0:
            return []
        else:
            return [[0]+b for b in BB(n-1,size)] + [[1]+b for b in BB(n-1,size-1)]

def PP(A,size = None):
    """
    Subsets of A (which should be a set or list).  If size is specified, only subsets
    of that size are returned.
    """
    A = sorted(list(A))
    n = len(A)
    if size is None:
        P = [set()]
        A.reverse()
        for a in A:
            P.extend([p.union({a}) for p in P])
        return P
    if size == 0:
        return [[]]
    L = [[j] for j in range(n-size+1)]
    for i in range(size-1):
        L = [l+[j] for l in L for j in range(l[i]+1,n-size+i+2)]
    return [{A[j] for j in l} for l in L]


def FF(A,size = None):
    """
    Lists of distinct elements of A (which should be a set or list).
    If size is specified, then lists of that size are returned.  If
    size is not specified, then lists of length equal to the size of A
    are returned: each such least must contain every element of A
    precisely once.
    """
    A = sorted(list(A))
    if size == None:
        size = len(A)
    L = [[]]    
    for i in range(size):
        L = [l+[a] for l in L for a in A if a not in l]
    return L
