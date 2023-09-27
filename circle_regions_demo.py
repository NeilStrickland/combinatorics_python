import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, PathPatch, Path

class circle_regions_demo:
    '''A demonstration of the number of regions created by drawing lines between n points on a circle'''
    rc = 0.02

    def __init__(self,angle):
        '''angle is a list of n numbers between 0 and 1, representing the angles of the points on the circle'''
        self.angle = sorted([t - np.floor(t) for t in angle])
        self.n = len(self.angle)
        self.S2 = [(i,j) for j in range(0,self.n) for i in range(0,j)]
        self.S4 = [(i,j,k,l) for l in range(0,self.n) for k in range(0,l) for j in range(0,k) for i in range(0,j)]
        self.outer_point = [np.array([np.cos(2*np.pi*t),np.sin(2*np.pi*t)]) for t in self.angle]
        self.normal = {}
        self.cutoff = {}
        for i in range(self.n-1):
            for j in range(i+1,self.n):
                u = np.array([self.outer_point[j][1]-self.outer_point[i][1],self.outer_point[i][0]-self.outer_point[j][0]])
                u = u / np.sqrt(u[0]*u[0] + u[1]*u[1])
                if u[0] * (1 - self.outer_point[i][0]) - u[1] * self.outer_point[i][1] < 0:
                    u = -u
                    self.normal[i,j] = u
                    self.cutoff[i,j] = u[0] * self.outer_point[i][0] + u[1] * self.outer_point[i][1]    

        self.crossing = {}
        U = [[self.outer_point[i],[i]] for i in range(self.n)]

        for i in range(self.n-3):
            for j in range(i+1,self.n-2):
                for k in range(j+1,self.n-1):
                    for l in range (k+1,self.n):
                        u0 = self.outer_point[i]
                        u1 = self.outer_point[j]
                        u2 = self.outer_point[k]
                        u3 = self.outer_point[l]
                        t = np.matmul(np.linalg.inv(np.transpose(np.array([u0-u2,u3-u1]))),u3-u2)
                        v = t[0]*u0 + (1-t[0])*u2
                        U.append([v,[i,j,k,l]])
                        self.crossing[i,j,k,l] = v

        self.all_points = U
        epsilon = 10. ** (-6)

        self.corners = {}
        self.centroid = {}
        for i in range(self.n-3):
            for j in range(i+1,self.n-2):
                for k in range(j+1,self.n-1):
                    for l in range (k+1,self.n):
                        v = self.crossing[i,j,k,l]
                        x1 = -self.normal[i,k]
                        x2 = -self.normal[j,l]

                        U1 = [u for u in U if u[1] != [i,j,k,l]]

                        for p in range(self.n-1):
                            for q in range(p+1,self.n):
                                w = self.normal[p,q]
                                c = self.cutoff[p,q]
                                if {p,q} == {i,k} or {p,q} == {j,l} or self.dp(v,w) < c:
                                    U1 = [u for u in U1 if self.dp(u[0],w) <= c + epsilon]
                                else:
                                    U1 = [u for u in U1 if self.dp(u[0],w) >= c - epsilon]

                        U1 = [[np.arctan2(self.dp(u[0] - v,x2),self.dp(u[0] - v,x1)),u[0],u[1]] for u in U1]
                        U1.sort(key = lambda a : a[0])
                        U1 = [[u[1],u[2]] for u in U1]
                        self.corners[i,j,k,l] = U1
                        m = len(U1)
                        w = (v + sum(u[0] for u in U1))/(m+1)
                        self.centroid[i,j,k,l] = w

    def dp(self,u,v):
        '''The dot product of two vectors u and v'''
        if len(u) != len(v):
            print(u)
            print(v)
            raise ValueError("u and v must be of the same length")
        return sum([x*y for x,y in zip(u,v)])

    def make_plots(self):
        '''Create the plots for the circle regions demo

        Each plot is a matplotlib patch or list of patches.  These can be added to a 
        matplotlib figure using the add_patch method of the figure's axes object.  
        '''
        self.circle_plot = [Circle((0,0),radius=1,edgecolor='black',facecolor='none'),
                            PathPatch(Path([(0.95,0.00),(1.05,0.00)]),color='black')]

        self.outer_point_plot = {}
        for i in range(self.n): 
            self.outer_point_plot[i] = Circle(self.outer_point[i],radius=self.rc,color='red')

        self.line_plot = {}
        for i in range(self.n-1):
            for j in range(i+1,self.n):
                self.line_plot[i,j] = PathPatch(Path([self.outer_point[i],self.outer_point[j]]),color="blue")

        self.crossing_plot = {}
        self.corners_plot = {}

        for i in range(self.n-3):
            for j in range(i+1,self.n-2):
                for k in range(j+1,self.n-1):
                    for l in range (k+1,self.n):
                        v = self.crossing[i,j,k,l]
                        self.crossing_plot[i,j,k,l] = Circle(v,radius=self.rc,color='magenta')
                        p = [v]
                        p.extend([u[0] for u in self.corners[i,j,k,l]])
                        self.corners_plot[i,j,k,l] = Polygon(p,color='yellow')

        self.outer_points_plot = [Circle(u,radius=self.rc,color='slategrey') for u in self.outer_point]
        self.crossings_plot = [Circle(self.crossing[ijkl],radius=self.rc,color='slategrey') for ijkl in self.S4]
        self.lines_plot = [PathPatch(Path([self.outer_point[i],self.outer_point[j]]),color='slategrey') 
                           for i in range(self.n-1) for j in range(i+1,self.n)]
        self.all_corners_plot = [self.corners_plot[S] for S in self.S4]

        F = []
        F.extend(self.circle_plot)
        F.extend(self.outer_points_plot)
        F.extend(self.crossings_plot)
        F.extend(self.lines_plot)
        self.full_plot = F

    def show_full_plot(self):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xlim(-1.1,1.1)
        ax.set_ylim(-1.1,1.1)
        ax.set_axis_off()
        self.make_plots()
        for p in self.full_plot:
            ax.add_patch(p)
