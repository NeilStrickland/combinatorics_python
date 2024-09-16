from basic import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, PathPatch, Path

def fat_point(u,ax,**kwargs):
    return ax.plot([u[0]],[u[1]],'o',**kwargs)

def triangle_shear(xy): 
    return [xy[0]+xy[1]/2,np.sqrt(3.)/2*xy[1]]

def triangle_point(u,ax,**kwargs): 
    return ax.plot(*triangle_shear(u),'.',**kwargs)

def triangle_text(u,s,ax,**kwargs): 
    return ax.text(*triangle_shear(u),s,horizontalalignment='center',verticalalignment='center',**kwargs)

def triangle_line(u,v,ax,**kwargs): 
    tu = triangle_shear(u)
    tv = triangle_shear(v)
    return ax.plot([tu[0],tv[0]],[tu[1],tv[1]],**kwargs)

def triangle_polygon(uu,ax,**kwargs):
    su = list(map(triangle_shear,uu))
    tu = list(map(tuple,su))
    return ax.add_patch(Polygon(tu,**kwargs))

def triangle_grid(n, ax=None, with_numbers=False, with_stripes=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_aspect('equal')
        ax.axis('off')
    for i in range(n):
        if with_stripes:
            triangle_line([i+1,0],[  0,i+1],ax,color="red",linewidth=3)
        else:
            triangle_line([i+1,0],[  0,i+1],ax,color="slategrey")
        triangle_line([  i,0],[  i,n-i],ax,color="slategrey")
        triangle_line([  0,i],[n-i,  i],ax,color="slategrey")
    if with_numbers:
        for i in range(1,n+1):
            for j in range(n+1-i):
                triangle_text([i-0.67,j+0.33],str(i),ax,color='black')
    return ax

def triangle_subtriangle(n,S,ax=None):
    if ax is None:
        ax = triangle_grid(n)
    if not isinstance(S,list) and len(S) == 3 and min(S) >= 0 and max(S) <= n+1:
        raise ValueError("S must be a list of three integers between 0 and n+1")
    [i,j,k] = sorted(S)
    u = [k-i-1,i]
    v = [j-i-1,i]
    w = [j-i-1,k-j+i]
    return triangle_polygon([u,v,w],ax,edgecolor='blue',facecolor='none',linewidth=3)

def triangle_count_picture(n,S,ax=None):
    if ax is None:
        ax = triangle_grid(n)
    if not isinstance(S,list) and len(S) == 3 and min(S) >= 0 and max(S) <= n+1:
        raise ValueError("S must be a list of three integers between 0 and n+1")
    [i,j,k] = sorted(S)
    u = [k-i-1,i]
    v = [j-i-1,i]
    w = [j-i-1,k-j+i]
    triangle_grid(n,ax)
    triangle_line([-1,0],[-1,n+1],ax,color="slategrey",linestyle='dotted')
    triangle_point([-1,i],ax)
    triangle_point([-1,j],ax)
    triangle_point([-1,k],ax)
    triangle_line([-1,i],u,ax,color='red')
    triangle_line([-1,j],v,ax,color='green')
    triangle_line([-1,k],u,ax,color='blue')
    triangle_line(v,w,ax,color='magenta')
    triangle_polygon([u,v,w],ax,color='yellow')
    return ax
