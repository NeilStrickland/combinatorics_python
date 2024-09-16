import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, PathPatch, Path

def is_nat_subset(k,n,S):
    """
    Check if S is a subset of {1,...,n} of size k.
    """
    if not isinstance(n,int) or not isinstance(k,int) or n < 0 or k < 0:
        return False
    return isinstance(S,set) and len(S) == k and \
        all([(isinstance(i,int) and i >= 1 and i <= n) for i in S])
    
def is_pos_sol(k,n,x):
    """
    Check if x is a solution to x_1 + ... + x_k = n with x_i > 0.
    """
    if not isinstance(n,int) or not isinstance(k,int) or n < 0 or k < 0:
        return False
    return isinstance(x,list) and len(x) == k and \
        all([isinstance(xi,int) and xi > 0 for xi in x]) and sum(x) == n

def list_pos_sols(k,n):
    """
    List all solutions to x_1 + ... + x_k = n with x_i > 0.
    """
    if not isinstance(n,int) or not isinstance(k,int) or n < 0 or k < 0:
        raise ValueError("n and k must be positive integers")
    if n < k or (k == 0 and n > 0):
        return []
    elif k == 1:
        return [[n]]
    else:
        return [[i]+s for i in range(1,n+1) for s in list_pos_sols(k-1,n-i)]
    
def count_pos_sols(k,n):
    """
    Count the number of solutions to x_1 + ... + x_k = n with x_i > 0.
    """
    if not isinstance(n,int) or not isinstance(k,int) or n < 0 or k < 0:
        raise ValueError("n and k must be positive integers")
    if n < k or (k == 0 and n > 0) or (n == 0 and k > 0):
        return 0
    return math.comb(n-1,k-1)

def set_to_pos_sol(k,n,S):
    """
    Proposition prop-sols-pos in the notes shows that there is a bijection 
    between the set of solutions to x_1 + ... + x_k = n with x_i > 0 and
    the set of subsets of size k-1 in {1,...,n-1}.  This function returns
    the solution corresponding to a given subset.
    """
    if not isinstance(n,int) or not isinstance(k,int) or n < 0 or k < 0:
        raise ValueError("n and k must be positive integers")
    if not is_nat_subset(k-1,n-1,S):
        raise ValueError(f"S must be a subset of size {k-1} in {{1,...,{n-1}}}")
    l = [0] + sorted(list(S)) + [n]
    return [l[i]-l[i-1] for i in range(1,k+1)]

def pos_sol_to_set(k,n,x):
    """
    Proposition prop-sols-pos in the notes shows that there is a bijection 
    between the set of solutions to x_1 + ... + x_k = n with x_i > 0 and
    the set of subsets of size k-1 in {1,...,n-1}.  This function returns
    the subset corresponding to a given solution.
    """
    if not isinstance(n,int) or not isinstance(k,int) or n < 0 or k < 0:
        raise ValueError("n and k must be positive integers")
    if not is_pos_sol(k,n,x):
        raise ValueError(f"x must be a list of {k} positive integers summing to {n}")
    S = [x[0]]
    for i in range(1,k-1):
        S.append(S[-1]+x[i])
    return set(S)

def is_nonneg_sol(k,n,x):
    """
    Check if x is a solution to x_1 + ... + x_k = n with x_i >= 0.
    """
    if not isinstance(n,int) or not isinstance(k,int) or n < 0 or k < 0:
        return False
    return isinstance(x,list) and len(x) == k and \
        all([isinstance(xi,int) and xi >= 0 for xi in x]) and sum(x) == n

def list_nonneg_sols(k,n):
    """
    List all solutions to x_1 + ... + x_k = n with x_i >= 0.
    """
    if n < 0:
        return []
    elif k == 1:
        return [[n]]
    else:
        return [[i]+s for i in range(n+1) for s in list_nonneg_sols(k-1,n-i)]

def count_nonneg_sols(k,n):
    """
    Count the number of solutions to x_1 + ... + x_k = n with x_i >= 0.
    """
    if not isinstance(n,int) or not isinstance(k,int) or n < 0 or k < 0:
        raise ValueError("n and k must be positive integers")
    if k == 0:
        return 1 if n == 0 else 0
    else:
        return math.comb(n+k-1,k-1)

def set_to_nonneg_sol(k,n,S):
    """
    Proposition prop-sols in the notes shows that there is a bijection 
    between the set of solutions to x_1 + ... + x_k = n with x_i >= 0 and
    the set of subsets of size k-1 in {1,...,n+k-1}.  This function returns
    the solution corresponding to a given subset.
    """
    if not isinstance(n,int) or not isinstance(k,int) or n < 0 or k < 0:
        raise ValueError("n and k must be positive integers")
    if not is_nat_subset(k-1,n+k-1,S):
        raise ValueError(f"S must be a subset of size {k-1} in {{0,...,{n+k-1}}}")
    l = [0] + sorted(list(S)) + [n+k]
    return [l[i]-l[i-1]-1 for i in range(1,k+1)]

def nonneg_sol_to_set(k,n,x):
    """
    Proposition prop-sols in the notes shows that there is a bijection 
    between the set of solutions to x_1 + ... + x_k = n with x_i >= 0 and
    the set of subsets of size k-1 in {1,...,n+k-1}.  This function returns
    the subset corresponding to a given solution.
    """
    if not isinstance(n,int) or not isinstance(k,int) or n < 0 or k < 0:
        raise ValueError("n and k must be positive integers")
    if not is_nonneg_sol(k,n,x):
        raise ValueError(f"x must be a list of {k} nonnegative integers summing to {n}")
    S = [x[0]+1]
    for i in range(1,k-1):
        S.append(S[-1]+x[i]+1)
    return set(S)

def is_grid_point(n,m,p):
    """
    Returns True iff p = (i,j) for some integers i, j with 0 <= i <= n and 0 <= j <= m.
    """
    return isinstance(p,tuple) and len(p) == 2 and \
        isinstance(p[0],int) and isinstance(p[1],int) and \
        p[0] >= 0 and p[0] <= n and p[1] >= 0 and p[1] <= m

def is_horizontal_step(a, b):
    return a[1] == b[1] and a[0] == b[0] - 1

def is_vertical_step(a, b):
    return a[0] == b[0] and a[1] == b[1] - 1

def is_grid_route(n,m,r):
    """
    Check if r is a route from (0,0) to (n,m) using only steps (1,0) and (0,1).
    """
    if not isinstance(n,int) or not isinstance(m,int) or n < 0 or m < 0:
        return False
    return isinstance(r,list) and len(r) == n+m+1 and \
        all([is_grid_point(n,m,p) for p in r]) and \
        r[0] == (0,0) and r[-1] == (n,m) and \
        all([(is_horizontal_step(r[i],r[i+1]) or is_vertical_step(r[i],r[i+1])) for i in range(n+m)])

def list_grid_routes(n,m):
    """
    List all routes from (0,0) to (n,m) using only steps (1,0) and (0,1).
    """
    if n == 0:
        return [[(0,j) for j in range(m+1)]]
    elif m == 0:
        return [[(i,0) for i in range(n+1)]]
    else:
        return [r+[(n,m)] for r in (list_grid_routes(n-1,m) + list_grid_routes(n,m-1))]
    
def count_grid_routes(n,m):
    """
    Count the number of routes from (0,0) to (n,m) using only steps (1,0) and (0,1).
    """
    if not isinstance(n,int) or not isinstance(m,int) or n < 0 or m < 0:
        raise ValueError("n and m must be nonnegative integers")
    return math.comb(n+m,n)

def set_to_grid_route(n,m,S):
    """
    Proposition prop-routes-grid in the notes shows that there is a bijection 
    between the set of routes from (0,0) to (n,m) using only steps (1,0) and (0,1)
    and the set of subsets of size n in {1,...,n+m}.  This function returns
    the route corresponding to a given subset.
    """
    if not isinstance(n,int) or not isinstance(m,int) or n < 0 or m < 0:
        raise ValueError("n and m must be nonnegative integers")
    if not is_nat_subset(n,n+m,S):
        raise ValueError(f"S must be a subset of size {n} in {{1,...,{n+m}}}")
    r = [(0,0)]
    for i in range(n+m):
        if i+1 in S:
            r.append((r[-1][0]+1,r[-1][1]))
        else:
            r.append((r[-1][0],r[-1][1]+1))
    return r

def grid_route_to_set(n,m,r):
    if not isinstance(n,int) or not isinstance(m,int) or n < 0 or m < 0:
        raise ValueError("n and m must be nonnegative integers")
    if not is_grid_route(n,m,r):
        raise ValueError("r must be a route from (0,0) to (n,m)")
    return {i for i in range(1,n+m+1) if is_horizontal_step(r[i-1],r[i])}     

def show_grid(n,m, ax=None):
    """
    Display an n by m grid.
    """
    nm = max(n,m)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10*(n+2)/(nm+2),10*(m+2)/(nm+2)))
        ax.set_xlim(-1,n+1)
        ax.set_ylim(-1,m+1)
        ax.set_axis_off()
    ax.vlines(range(n+1),0,m,color="slategrey")
    ax.hlines(range(m+1),0,n,color="slategrey")
    return ax

def show_grid_route(r,n=None,m=None,ax=None):
    """
    Display a route as a grid.
    """
    if n is None:
        n = max([x for (x,y) in r])
    if m is None:
        m = max([y for (x,y) in r])
    nm = max(n,m)
    if ax is None:
        ax = show_grid(n,m)
    for k in range(len(r)-1):
        if (r[k][0] == r[k+1][0]):
            c = 'blue'
        else:
            c = 'red'
        ax.plot([r[k][0],r[k+1][0]],[r[k][1],r[k+1][1]],color=c,linewidth=5)
    return ax