import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, PathPatch, Path

class knights_tour:
    steps = [
     [ 0, 5, 2,57,62,15,18,21],
     [ 3,56,63,14, 9,20,61,16],
     [ 6, 1, 4,43,58,17,22,19],
     [55,44,27, 8,13,10,39,60],
     [28, 7,42,45,40,59,12,23],
     [51,54,49,26,11,36,33,38],
     [48,29,52,41,46,31,24,35],
     [53,50,47,30,25,34,37,32]
    ]

    def __init__(self):
        self.pos = {}
        self.opos = {}
        for i in range(8):
            for j in range(8):
                self.pos[self.steps[i][j]] = [i,j]
                self.opos[self.steps[i][j]] = [i+0.5,j+0.5]
        self.pos[64] = self.pos[0]
        self.opos[64] = self.opos[0] 

    def is_knight_move(self,u,v):
        w = sorted([abs(u[i] - v[i]) for i in range(2)])
        return w == [1,2]
    
    def check(self):
        return all([self.is_knight_move(self.pos[i],self.pos[i+1]) for i in range(64)])
    
    def make_plots(self):
        self.grid = []
        self.grid.extend([PathPatch(Path([(i,0),(i,8)]),color="slategrey") for i in range(9)])
        self.grid.extend([PathPatch(Path([(0,i),(8,i)]),color="slategrey") for i in range(9)])
        self.shaded_grid = []
        self.shaded_grid.extend([PathPatch(Path([(i,0),(i,8)]),color="slategrey") for i in range(9)])
        self.shaded_grid.extend([PathPatch(Path([(0,i),(8,i)]),color="slategrey") for i in range(9)])
        self.shaded_grid.extend([Rectangle((2*i,2*j+1),1,1,color="slategrey") for i in range(4) for j in range(4)])
        self.shaded_grid.extend([Rectangle((2*i+1,2*j),1,1,color="slategrey") for i in range(4) for j in range(4)])
        self.tour_plot = []
        self.tour_plot.extend([PathPatch(Path([(i,0),(i,8)]),color="slategrey") for i in range(9)])
        self.tour_plot.extend([PathPatch(Path([(0,i),(8,i)]),color="slategrey") for i in range(9)])
        self.tour_plot.extend([Rectangle((2*i,2*j+1),1,1,color="slategrey") for i in range(4) for j in range(4)])
        self.tour_plot.extend([Rectangle((2*i+1,2*j),1,1,color="slategrey") for i in range(4) for j in range(4)])
        self.tour_plot.extend([PathPatch(Path([self.opos[i],self.opos[i+1]]),color="red") for i in range(64)])

    def show_tour_plot(self):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xlim(-0.5,8.5)
        ax.set_ylim(-0.5,8.5)
        ax.set_axis_off()
        self.make_plots()
        for p in self.tour_plot:
            ax.add_patch(p)
