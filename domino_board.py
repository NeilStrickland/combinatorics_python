import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, PathPatch, Path

class domino_board:
    def __init__(self,S):
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
                if (i,j) in self.squares:
                    self.shaded_plot.append(Rectangle((i,j),1,1,color="black"))
                elif (i,j) in self.black_squares:
                    self.shaded_plot.append(Rectangle((i,j),1,1,color="slategrey"))
    
    def domino(self,i,j,o):
        if o == 0:
            return Rectangle((i,j),2,1,color="red")
        else:
            return Rectangle((i,j),1,2,color="red")