from basic import *
import math

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

def is_binseq(n,size,x):
    """
    Check if x is a binary sequence of length n with k ones
    If n is None, then the length is not checked
    If size is None, then the number of ones is not checked
    """
    if not (isinstance(x,list) and all([y in [0,1] for y in x])):
        return False
    if not(n is None or len(x) == n):
        return False
    if not(size is None or sum(x) == size):
        return False
    return True

def list_binseqs(n,size = None):
    """
    Returns the list of all binary sequences of length n.  
    If size is specified, only sequences with that number of ones are returned.
    """
    if not isinstance(n,int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if size is None:
        B = [[]]
        for _ in range(n):
            B = [b+[x] for b in B for x in range(2)]
        return B
    else:
        if not isinstance(size,int) or size < 0:
            raise ValueError("size must be a nonnegative integer")
        if size == 0:
            return [[0 for _ in range(n)]]
        elif n == 0:
            return []
        else:
            return [[0]+b for b in list_binseqs(n-1,size)] + \
                   [[1]+b for b in list_binseqs(n-1,size-1)]

def count_binseqs(n,size = None):
    """
    Returns the number of binary sequences of length n.  
    If size is specified, only sequences with that number of ones are counted.
    """
    if not isinstance(n,int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if size is None:
        return 2 ** n
    elif not isinstance(size,int) or size < 0:
        raise ValueError("size must be a nonnegative integer")
    elif size > n:
        return 0
    else:
        return math.comb(n,size)

def is_subset(A,size,B):
    """
    Check if B is a subset of A of the specified size.  
    If size is None, then the size is not checked.
    Here A and B should be sets, or objects that can be converted to 
    sets by the to_set() function.
    """
    A = to_set(A)
    B = to_set(B)
    if not(B.issubset(A)):
        return False
    if size is not None and len(B) != size:
        return False
    return True

def list_subsets(A,size = None):
    """
    List of subsets of A. If size is specified, only subsets of that size are returned.
    Here A should be a set, or an object that can be converted to 
    a set by the to_set() function.
    """
    if size is not None and size < 0:
        return []
    if size == 0:
        return [[]]
    A = sorted(to_list(A))
    n = len(A)
    if size is None:
        P = [set()]
        A.reverse()
        for a in A:
            P.extend([p.union({a}) for p in P])
        return P
    elif not isinstance(size,int) or size < 0:
        raise ValueError("size must be a nonnegative integer")
    L = [[j] for j in range(n-size+1)]
    for i in range(size-1):
        L = [l+[j] for l in L for j in range(l[i]+1,n-size+i+2)]
    return [{A[j] for j in l} for l in L]

def count_subsets(A,size = None):
    """
    Number of subsets of A.  If size is specified, only subsets of 
    that size are counted.
    Here A should be a set, or an object that can be converted to 
    a set by the to_set() function.
    """
    n = len(to_list(A))
    return count_binseqs(n,size)

def binseq_to_nat_subset(n, x):
    return {i for i in range(1,n+1) if x[i-1] == 1}

def nat_subset_to_binseq(n, S):
    return [1 if i in S else 0 for i in range(1,n+1)]

def list_distinct_seqs(A,size = None):
    """
    List of lists of distinct elements of A.
    If size is specified, then lists of that length are returned.  If
    size is not specified, then lists of length equal to the size of A
    are returned: each such least must contain every element of A
    precisely once.
    Here A should be a set, or an object that can be converted to 
    a set by the to_set() function.
    """
    A = sorted(to_list(to_set(A)))
    if size is None:
        size = len(A)
    elif not isinstance(size,int) or size < 0:
        raise ValueError("size must be a nonnegative integer")
    if size > len(A):
        return []
    L = [[]]    
    for i in range(size):
        L = [l+[a] for l in L for a in A if a not in l]
    return L

def count_distinct_seqs(A,size = None):
    """
    Lists of distinct elements of A (which should be a set or list).
    If size is specified, then lists of that size are returned.  If
    size is not specified, then lists of length equal to the size of A
    are returned: each such list must contain every element of A
    precisely once.
    """
    if isinstance(A,int) and A >= 0:
        n = A
    else:
        n = len(to_set(A))
    if size is None:
        size = n
    elif not isinstance(size,int) or size < 0:
        raise ValueError("size must be a nonnegative integer")
    if size > n:
        return 0
    m = 1
    for i in range(size):
        m *= n-i
    return m

def is_gappy_binseq(n,size,x):
    """
    Check if x is a binary sequence of length n with k ones
    If n is None, then the length is not checked
    If size is None, then the number of ones is not checked
    """
    if not(is_binseq(n,size,x)):
        return False
    if any([x[i] == 1 and x[i+1] == 1 for i in range(len(x)-1)]):
        return False
    return True

def binseq_to_gappy_binseq(x):
    y = []
    started = False
    for i in range(len(x)):
        if x[i] == 0:
            y.append(0)
        else:
            if started:
                y.append(0)
            y.append(1)
            started = True
    return y

def gappy_binseq_to_binseq(x):
    y = []
    n = len(x)
    i = 0
    started = False
    for i in range(n):
        if i < n-1 and x[i+1] == 1:
            if not started:
                y.append(0)
                started = True
        else:
            y.append(x[i])
    return y

def list_gappy_binseqs(n,size):
    return [binseq_to_gappy_binseq(x) for x in list_binseqs(n+1-size,size)]

def count_gappy_binseqs(n,size):
    return math.comb(n+1-size,size)

def is_gappy_subset(n,size,S):
    if not(is_subset(n,size,S)):
        return False
    if any([i+1 in S for i in S]):
        return False
    return True

def list_gappy_sets(n,size):
    return [binseq_to_nat_subset(n,x) for x in list_gappy_binseqs(n,size)]

def count_gappy_sets(n,size):
    return math.comb(n+1-size,size)

def fibonacci(n):
    if not isinstance(n,int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if n <= 1:
        return 1
    a = 1
    b = 1
    for _ in range(n-1):
        a,b = b,a+b
    return b

def fibonacci_alt(n):
    if not isinstance(n,int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    return sum([math.comb(n-i,i) for i in range(n//2+1)])