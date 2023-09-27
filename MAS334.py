import numpy as np
from primePy import primes
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, PathPatch, Path

def prime_root(n):
    if not(isinstance(n, int) and n > 0):
        raise ValueError("n must be an integer greater than 0")
    factors = primes.factors(n)
    if len(factors) == 0:
        return None
    if factors[:-1] != factors[1:]:
        return None
    return factors[0]

def dp(u,v):
    if len(u) != len(v):
        raise ValueError("u and v must be of the same length")
    return sum([x*y for x,y in zip(u,v)])

def random_element_of(X):
    if isinstance(X,int):
        return random.randint(1,X)
    if isinstance(X,set):
        X = list(X)
    if not isinstance(X,list):
        raise ValueError("X must be an integer, a set or a list")
    return random.choice(X)

def random_subset_of(X,n=False):
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

def fat_point(ax,u,**kwargs):
    return ax.plot([u[0]],[u[1]],'o',**kwargs)

def triangle_shear(xy): 
    return [xy[0]+xy[1]/2,np.sqrt(3.)/2*xy[1]]

def triangle_point(ax,u,**kwargs): 
    return ax.plot(*triangle_shear(u),'.',**kwargs)

def triangle_line(ax,u,v,**kwargs): 
    tu = triangle_shear(u)
    tv = triangle_shear(v)
    return ax.plot([tu[0],tv[0]],[tu[1],tv[1]],**kwargs)

def triangle_polygon(ax,uu,**kwargs):
    su = list(map(triangle_shear,uu))
    tu = list(map(tuple,su))
    ax.add_patch(Polygon(tu,**kwargs))

def triangle_grid(ax,n):
    for i in range(n):
        triangle_line(ax,[0,i],[n-i,i],color="slategrey")
        triangle_line(ax,[i,0],[i,n-i],color="slategrey")
        triangle_line(ax,[i+1,0],[0,i+1],color="slategrey")

def triangle_count_picture(ax,n,S):
    if not isinstance(S,list) and len(S) == 3 and min(S) >= 0 and max(S) <= n+1:
        raise ValueError("S must be a list of three integers between 0 and n+1")
    [i,j,k] = sorted(S)
    u = [k-i-1,i]
    v = [j-i-1,i]
    w = [j-i-1,k-j+i]
    triangle_grid(ax,n)
    triangle_line(ax,[-1,0],[-1,n+1],color="slategrey",linestyle='dotted')
    triangle_point(ax,[-1,i])
    triangle_point(ax,[-1,j])
    triangle_point(ax,[-1,k])
    triangle_line(ax,[-1,i],u,color='red')
    triangle_line(ax,[-1,j],v,color='green')
    triangle_line(ax,[-1,k],u,color='blue')
    triangle_line(ax,v,w,color='magenta')
    triangle_polygon(ax,[u,v,w],color='yellow')

