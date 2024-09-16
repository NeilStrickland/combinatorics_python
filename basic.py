import matplotlib.pyplot as plt
from primePy import primes
import random

def get_ax(ax = None):
    """
    If ax is None, then get_ax returns a new figure and axis.
    If ax is a matplotlib axis, then get_ax returns ax.
    """
    if ax == None:
        fig, ax = plt.subplots()
    return ax

def prime_root(n):
    """
    If n = p ** k for a prime p and an integer k > 0, then prime_root(n) returns p
    If n is not of this form, then prime_root(n) returns None
    """
    if not(isinstance(n, int) and n > 0):
        raise ValueError("n must be an integer greater than 0")
    factors = primes.factors(n)
    if len(factors) == 0:
        return None
    if factors[:-1] != factors[1:]:
        return None
    return factors[0]

def dp(u,v):
    """
    If u and v are lists of the same length, then dp(u,v) returns the dot product of u and v
    """
    if len(u) != len(v):
        raise ValueError("u and v must be of the same length")
    return sum([x*y for x,y in zip(u,v)])

def random_element_of(X):
    """
    If X is a list or a set, then random_element_of(X) returns a random element of X
    If X is a positive integer, then random_element_of(X) returns a random integer between 1 and X
    """
    if isinstance(X,int):
        return random.randint(1,X)
    if isinstance(X,set):
        X = list(X)
    if not isinstance(X,list):
        raise ValueError("X must be an integer, a set or a list")
    return random.choice(X)

def random_subset_of(X,n=False):
    """
    Returns a random subset of X of size n
    If n is not specified, then the size of the subset is chosen randomly
    (This means that very large and very small subsets are more likely to be chosen
    than would be the case if we chose the subset uniformly at random.)
    If X is an integer, then it is interpreted as the set {1,2,...,X}
    """
    if isinstance(X,int):
        X = list(range(1,X+1))
    if isinstance(X,set):
        X = list(X)
    if not isinstance(X,list):
        raise ValueError("X must be an integer, a set or a list")
    if n == False:
        n = random.randint(0,len(X))
    if n > len(X):
        raise ValueError("n must be less than or equal to the size of X")
    return set(random.sample(X,n))

def to_list(x):
    """
    Convert x to a list.  If x is a nonnegative integer, then to_list(x) returns
    the list [1,2,...,x].  Otherwise, to_list(x) uses the built in list() function,
    which converts most list-like things (such as sets, tuples or numpy arrays) 
    to lists.
    """
    if isinstance(x,int) and x >= 0:
        return list(range(1, x+1))
    else:
        return list(x)

def to_set(x):
    """
    Convert x to a set.  If x is a nonnegative integer, then to_set(x) returns
    the set {1,2,...,x}.  Otherwise, to_set(x) uses the built in set() function,
    which converts most set-like things (such as lists, tuples or numpy arrays) 
    to sets.
    """
    if isinstance(x,int) and x >= 0:
        return set(range(1, x+1))
    else:
        return set(x)

