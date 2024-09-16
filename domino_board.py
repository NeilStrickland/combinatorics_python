import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, PathPatch, Path, FancyBboxPatch

class domino_board:
    def __init__(self,S,complement=False):
        if isinstance(S,int):
            n = S
            S = [(i,j) for i in range(n) for j in range(n)]
        elif isinstance(S,tuple) and len(S) == 2 and all([isinstance(x,int) for x in S]):
            n,m = S
            S = [(i,j) for i in range(n) for j in range(m)]
        elif all([isinstance(s,tuple) and len(s) == 2 and \
                  all([(isinstance(x,int) and x >= 0) for x in s]) for s in S]):
            S = list(S)
        else:
            raise ValueError("Input must be a list of tuples of integers of length 2")
        if complement:
            n = 1 + max([s[0] for s in S])
            m = 1 + max([s[1] for s in S])
            S = [(i,j) for i in range(n) for j in range(m) if (i,j) not in S]
        self.squares = S
        self.black_squares = [s for s in S if (s[0]+s[1]) % 2 == 0]
        self.white_squares = [s for s in S if (s[0]+s[1]) % 2 == 1]
        self.num_black_squares = len(self.black_squares)
        self.num_white_squares = len(self.white_squares)
        self.num_squares = len(self.squares)
        self.width = 1 + max([s[0] for s in S])
        self.height = 1 + max([s[1] for s in S])

    def make_plots(self):
        self.grid_plot = []
        self.grid_plot.extend([PathPatch(Path([(i,0),(i,self.height)]),color="slategrey") for i in range(self.width+1)])
        self.grid_plot.extend([PathPatch(Path([(0,i),(self.width,i)]),color="slategrey") for i in range(self.height+1)])
        self.shaded_plot = []
        self.shaded_plot.extend([PathPatch(Path([(i,0),(i,self.height)]),color="slategrey") for i in range(self.width+1)])
        self.shaded_plot.extend([PathPatch(Path([(0,i),(self.width,i)]),color="slategrey") for i in range(self.height+1)])
        for i in range(self.width):
            for j in range(self.height):
                if (i,j) not in self.squares:
                    self.shaded_plot.append(Rectangle((i,j),1,1,color="black"))
                elif (i,j) in self.black_squares:
                    self.shaded_plot.append(Rectangle((i,j),1,1,color="slategrey"))
    
    def domino(self,i,j,o):
        e = 0.2
        p = 1.6
        q = 0.6
        if o == 0:
            return FancyBboxPatch((i+e,j+e),p,q,color="red",boxstyle="round,pad=0.1")
        else:
            return FancyBboxPatch((i+e,j+e),q,p,color="red",boxstyle="round,pad=0.1")
        
    def add_patches(self, ax, patches):
        for p in patches:
            ax.add_patch(p)
        return ax

    def show_bare_plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,10))
            ax.set_aspect('equal')
            ax.set_xlim(-1,self.width+1)
            ax.set_ylim(-1,self.height+1)
            ax.set_axis_off()
        return ax
    
    def show_grid_plot(self, ax=None):
        if ax is None:
            ax = self.show_bare_plot()
        self.add_patches(ax, self.grid_plot)
        return ax
    
    def show_shaded_plot(self, ax=None):
        if ax is None:
            ax = self.show_bare_plot()
        self.add_patches(ax, self.shaded_plot)
        return ax
    
    def add_domino(self, ax, i, j, o):
        ax.add_patch(self.domino(i,j,o))
        return ax
