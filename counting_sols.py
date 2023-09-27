import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, PathPatch, Path

def list_pos_sols(k,n):
    """
    List all solutions to x_1 + ... + x_k = n with x_i > 0.
    """
    if n < k:
        return []
    elif k == 1:
        return [[n]]
    else:
        return [[i]+s for i in range(1,n+1) for s in list_pos_sols(k-1,n-i)]
    
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
    
def show_grid_route(r):
    """
    Display a route as a grid.
    """
    n = max([x for (x,y) in r])
    m = max([y for (x,y) in r])
    nm = max(n,m)
    fig, ax = plt.subplots(figsize=(10*(n+2)/(nm+2),10*(m+2)/(nm+2)))
    ax.set_xlim(-1,n+1)
    ax.set_ylim(-1,m+1)
    ax.set_axis_off()
    for i in range(n+1):
        ax.add_patch(PathPatch(Path([(i,0),(i,m)]),color="slategrey"))
    for j in range(m+1):
        ax.add_patch(PathPatch(Path([(0,j),(n,j)]),color="slategrey"))
    for k in range(len(r)-1):
        if (r[k][0] == r[k+1][0]):
            c = 'blue'
        else:
            c = 'red'
        ax.add_patch(PathPatch(Path([r[k],r[k+1]]),color=c,linewidth=5))
