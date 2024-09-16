import numpy as np
import matplotlib.pyplot as plt

class Point:
    """
    Represents a point on a Venn diagram where two circles intersect.

    Attributes:
    index: int
        The index of the point in the Venn diagram.  A Venn diagram with two 
        circles has two points, with indices 0 and 1.  A Venn diagram with three
        circles has six points, with indices 0 to 5.

    coords: tuple
        The coordinates of the point in the Venn diagram.

    adjacent_regions: list
        A list of the indices of the regions adjacent to the point.

    inward_arcs: list
        A list of the indices of the arcs that terminate at the point.

    outward_arcs: list
        A list of the indices of the arcs that originate at the point.

    colour: str
        The colour of the point in the Venn diagram.
    """
    def __init__(self, index, coords=(0,0), adjacent_regions=None, inward_arcs=None, outward_arcs=None, colour='black'):
        self.index = index
        self.coords = coords
        self.adjacent_regions = [] if adjacent_regions is None else adjacent_regions
        self.inward_arcs = [] if inward_arcs is None else inward_arcs
        self.outward_arcs = [] if outward_arcs is None else outward_arcs
        self.adjacent_arcs = self.inward_arcs + self.outward_arcs
        self.colour = colour
        self.obj = None
        self.label = None

    def __str__(self):
        return f'Point {self.index} at {self.coords}'

    def draw(self, ax):
        self.obj = ax.plot(self.coords[0], self.coords[1], 'o', color=self.colour, markersize=7)

    def draw_label(self, ax, offset=(0,0)):
        self.label = ax.text(self.coords[0]+offset[0], self.coords[1]+offset[1], f'{self.index}', fontsize=12)

class Arc:
    """
    Represents an arc on a Venn diagram, i.e. a segment of one of the circles,
    between two of the intersection points.  Arcs are parametrised in an 
    anticlockwise direction, and this direction is used to determine the
    start and end point, and the regions to the left and right of the arc.

    Attributes:
    index: int
        The index of the arc in the Venn diagram.  A Venn diagram with two 
        circles has four arcs, with indices 0 to 3.  A Venn diagram with three
        circles has twelve arcs, with indices 0 to 11.

    type: str
        The type of the arc.  This can be 'inner', 'intermediate cw', 'intermediate acw', or 'outer'.

    centre: tuple
        The coordinates of the centre of the circle that the arc belongs to.

    radius: float
        The radius of the circle that the arc belongs to.

    min_angle: float
        The minimum angle of the arc, in radians.

    max_angle: float
        The maximum angle of the arc, in radians.

    start: int
        The index of the point where the arc starts.

    end: int
        The index of the point where the arc ends.

    left_region: int
        The index of the region to the left of the arc.

    right_region: int
        The index of the region to the right of the arc.

    colour: str
        The colour of the arc in the Venn diagram.
    """
    def __init__(self, index, type='', centre=(0,0), radius=1, min_angle=0, max_angle=0,
                 start=0, end=0, left_region=0, right_region=0, colour='black'):
        self.index = index
        self.type = type
        self.radius = radius
        self.centre = centre
        self.min_angle = min_angle
        self.max_angle = max_angle
        x0, y0 = centre
        midpoint_angle = (min_angle + max_angle) / 2
        self.midpoint_coords = (x0 + radius * np.cos(midpoint_angle), y0 + radius * np.sin(midpoint_angle))
        self.midpoint_direction = (-np.sin(midpoint_angle), np.cos(midpoint_angle))
        self.midpoint_normal = (-np.cos(midpoint_angle), -np.sin(midpoint_angle))
        self.start = start
        self.end = end
        self.adjacent_points = [self.start, self.end]
        self.left_region = left_region
        self.right_region = right_region
        self.adjacent_regions = [self.left_region, self.right_region]
        self.colour = colour
        ts = np.linspace(self.min_angle, self.max_angle, 100)
        self.xs = np.cos(ts) * self.radius + self.centre[0]
        self.ys = np.sin(ts) * self.radius + self.centre[1]
        self.obj = None
        self.midpoint_arrow = None
        self.label = None

    def __str__(self):
        return f'Arc {self.index} from {self.start} to {self.end}'

    def draw(self, ax):
        self.obj = ax.plot(self.xs, self.ys, '-', color=self.colour)

    def draw_midpoint_arrow(self, ax):
        self.midpoint_arrow = ax.arrow(self.midpoint_coords[0], self.midpoint_coords[1], 
                                       0.01 * self.midpoint_direction[0], 0.01 * self.midpoint_direction[1], 
                                       head_width=0.15, head_length=0.15, fc=self.colour, ec=self.colour)

    def draw_label(self, ax, offset=(0,0)):
        self.label = ax.text(self.midpoint_coords[0]+offset[0], self.midpoint_coords[1]+offset[1], 
                             f'{self.index}', fontsize=12)

class Region:
    """
    Represents a region on a Venn diagram, i.e. one of the areas enclosed by the arcs.
    For example, a Venn diagram with two circles (corresponding to sets X and Y) has
    four regions, corresponding to the sets X∩Y, X\Y, Y\X, and (X∪Y)^c.  The sets
    X and Y themselves do not count as regions.

    Attributes:
    index: int
        The index of the region in the Venn diagram.  A Venn diagram with two 
        circles has four regions, with indices 0 to 3.  A Venn diagram with three
        circles has eight regions, with indices 0 to 7.

    type: str
        The type of the region.  This can be 'inner', 'intermediate', 'outer' or 'exterior'.

    centre: tuple
        The coordinates of a point roughly in the middle of the region, where a label can be placed.

    adjacent_points: list
        A list of the indices of the points adjacent to the region.

    adjacent_oriented_arcs: list 
        A list of pairs of the indices of the arcs adjacent to the region, and the direction
        in which they are oriented.  The direction is 1 if the arc is oriented anticlockwise
        around the region, and -1 if the arc is oriented clockwise around the region.

    components: list
        The Venn diagram is based on a pair of sets X0, X1, or a triple of sets X0, X1, X2.
        For each region there is a set C such that the region is the intersection of the sets
        Xi for i in C and Xi^c for i not in C.  The components attribute is a list of the
        indices that lie in C.

    colour: str
        The colour of the region in the Venn diagram.
    """
    def __init__(self, index, type = '', centre=(0,0), adjacent_points = None, adjacent_oriented_arcs = None, components = None, colour = 'yellow'):
        """
        """
        self.index = index
        self.type = type
        self.centre = centre
        self.adjacent_points = [] if adjacent_points is None else adjacent_points
        self.adjacent_oriented_arcs = [] if adjacent_oriented_arcs is None else adjacent_oriented_arcs
        self.adjacent_arcs = [a[0] for a in adjacent_oriented_arcs]
        self.components = [] if components is None else components
        self.colour = colour
        self.obj = None
        self.centre_marker = None
        self.label = None

    def __str__(self):
        return f'Region {self.index} with components {self.components}'

    def set_xys(self, arcs):
        """
        """
        xys = []
        for a, d in self.adjacent_oriented_arcs:
            arc = arcs[a]
            if d == 1:
                xys.extend(zip(arc.xs, arc.ys))
            else:
                xys.extend(zip(arc.xs[::-1], arc.ys[::-1]))
        self.xs = np.array([xy[0] for xy in xys])
        self.ys = np.array([xy[1] for xy in xys])

    def draw(self, ax):
        self.obj = ax.fill(self.xs, self.ys, color=self.colour, edgecolor=None, closed=True)

    def draw_centre_marker(self, ax):
        self.centre_marker = ax.plot(self.centre[0], self.centre[1], 'o', color='black', markersize=5)

    def draw_label(self, ax, offset=(0,0)):
        self.label = ax.text(self.centre[0]+offset[0], self.centre[1]+offset[1], f'{self.index}', fontsize=12)

class Subset():
    """
    Represents a subset of a Venn diagram.  For a Venn diagram with two circles corresponding
    to subsets X and Y in a universal set U, there are 16 possible subsets, such as 
    X∩Y, X\Y, Y\X, U\X, U\Y, etc.  For a Venn diagram with three circles, there are 256
    possible subsets.

    Attributes:
    index: int
        The index of the subset in the Venn diagram.  A Venn diagram with two circles
        has 16 subsets, with indices 0 to 15.  A Venn diagram with three circles has
        256 subsets, with indices 0 to 255.

    name: str
        The name of the subset, in ASCII characters.  In the case of a Venn diagram with
        two circles, the basic sets are named X and Y.  In a 3-circle Venn diagram, the
        basic sets are named X, Y and Z.  The letters n, u, d and c are
        used for intersection, union, symmetric difference and complement, respectively.
        In a 3-circle Venn diagram, some of the subsets have no simple name, and are
        represented by a string of digits corresponding to the regions in the subset.

    latex_name: str
        The name of the subset, in LaTeX format.  In cases where the name is a string
        of digits, the LaTeX name is the same as the name.
    """

    def __init__(self, index, name, latex_name, regions):
        self.index = index
        self.name = name
        self.latex_name = latex_name
        self.regions = regions

    def __str__(self):
        return f'Subset {self.index} with name {self.name} and regions {self.regions}'

class Venn():
    """
    Represents a Venn diagram, which might have two or three circles.
    This class has functions for that are common to both two-circle and
    three-circle cases.  There are also subclasses Venn2 and Venn3 that
    are specific to the two-circle and three-circle cases.
    """
    def __init__(self):
        self.box_size = 4
        self.ax = None

    def get_ax(self):
        if self.ax is not None:
            return self.ax
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(-self.box_size, self.box_size)
        ax.set_ylim(-self.box_size, self.box_size)
        self.ax = ax
        return ax
    
    def draw_points(self, ax=None, markers=True, labels=False):
        if ax is None:
            ax = self.get_ax()
        for p in self.points:
            if markers:
                p.draw(ax)
            if labels:
                p.draw_label(ax)

    def draw_arcs(self, ax=None, labels=False, arrows=False):
        if ax is None:
            ax = self.get_ax()
        for a in self.arcs:
            a.draw(ax)
            if arrows:
                a.draw_midpoint_arrow(ax)
            if labels:
                a.draw_label(ax)
    
    def draw_regions(self, ax=None, patches=True, labels=False, markers=False):
        if ax is None:
            ax = self.get_ax()
        for r in self.regions:
            if patches:
                r.draw(ax)
            if labels:
                r.draw_label(ax)
            if markers:
                r.draw_centre_marker(ax)

    def draw_region(self, i, ax=None, patches=True, labels=False, markers=False):
        if ax is None:
            ax = self.get_ax()
        r = self.regions[i]
        if patches:
            r.draw(ax)
        if labels:
            r.draw_label(ax)
        if markers:
            r.draw_centre_marker(ax)

    def draw_subset(self, i, ax=None, patches=True, caption=True, labels=False):
        if ax is None:
            ax = self.get_ax()
        if isinstance(i, Subset):
            s = i
        elif isinstance(i, int):
            s = self.subsets[i]
        else:
            s = self.subsets_by_name[i]
        for j in s.regions:
            self.draw_region(j, ax, patches, labels)
        if caption:
            ax.text(0, 3.8, '$' + s.latex_name + '$', fontsize=12, ha='center')

class Venn2(Venn):
    def __init__(self):
        self.box_size = 4
        self.U_radius = 7/2
        self.points = []
        self.arcs = []
        self.regions = []
        self.subsets = []
        self.subsets_by_name = {}
        self.ax = None

        for i in range(2):
            p = Point(index=i, 
                      coords = (0,1.5 * (-1)**i), 
                      adjacent_regions = [i,3,1-i,2],
                      inward_arcs = [1-i,2+i],
                      outward_arcs = [i,3-i])
            self.points.append(p)

        arcs = {}
        for i in range(2):
            for j in range(2):
                m = (2/3*(1+j) + i) * np.pi
                w = 4/3 * np.pi if j == 1 else 2/3 * np.pi
                x0 = (-1)**i * np.sqrt(3)/2
                y0 = 0
                types = ['inner','outer']
                left_regions = [2,i]
                right_regions = [1-i,3]
                colours = ['red','blue']
                a = Arc(index=i+2*j, 
                        type = types[j],
                        centre = (x0, y0), 
                        radius = np.sqrt(3), 
                        min_angle = m, 
                        max_angle = m+w,
                        start = (i + j) % 2,
                        end = (i + j + 1) % 2,
                        left_region = left_regions[j],
                        right_region = right_regions[j],
                        colour = colours[j])
                arcs[a.index] = a

        self.arcs = [arcs[i] for i in range(4)]

        for i in range(2):
            r = Region(index=i,
                       type="outer",
                       centre=(2*(-1)**i,0),
                       adjacent_points=[0,1],
                       adjacent_oriented_arcs=[[i+2,1],[1-i,-1]],
                       components=[i])
            self.regions.append(r)

        r = Region(index=2,
                   type="centre",
                   centre=(0,0),
                   adjacent_points=[0,1],
                   adjacent_oriented_arcs=[[0,1],[1,1]],
                   components=[0,1])
        self.regions.append(r)

        r = Region(index=3,
                   type="exterior",
                   centre=(0,-9/4),
                   adjacent_points=[0,1],
                   adjacent_oriented_arcs=[[2,-1],[3,-1]],
                   components=[])
        self.regions.append(r)
        
        for i in range(4):
            self.regions[i].set_xys(self.arcs)

        r = self.regions[3]
        ts = np.linspace(np.pi/2, 5*np.pi/2, 200)
        xs = np.cos(ts) * self.U_radius
        ys = np.sin(ts) * self.U_radius
        r.xs = np.array(list(r.xs) + list(xs) + [r.xs[0]])
        r.ys = np.array(list(r.ys) + list(ys) + [r.ys[0]])

        self.subsets = [
            Subset(index =   0,name = "O", latex_name="\\emptyset", regions=[]),
            Subset(index =   1,name = "X\\Y", latex_name="X\\setminus Y", regions=[0]),
            Subset(index =   2,name = "Y\\X", latex_name="Y\\setminus X", regions=[1]),
            Subset(index =   3,name = "XnY", latex_name="X\\cap Y", regions=[2]),
            Subset(index =   4,name = "(XuY)c", latex_name="(X\\cup Y)^c", regions=[3]),
            Subset(index =   5,name = "XdY", latex_name="X\\Delta Y", regions=[0,1]),
            Subset(index =   6,name = "X", latex_name="X", regions=[0,2]),
            Subset(index =   7,name = "Yc", latex_name="Y^c", regions=[0,3]),
            Subset(index =   8,name = "Y", latex_name="Y", regions=[1,2]),
            Subset(index =   9,name = "Xc", latex_name="X^c", regions=[1,3]),
            Subset(index =  10,name = "(XdY)c", latex_name="(X\\Delta Y)^c", regions=[2,3]),
            Subset(index =  11,name = "XuY", latex_name="X\\cup Y", regions=[0,1,2]),
            Subset(index =  12,name = "(XnY)c", latex_name="(X\\cap Y)^c", regions=[0,1,3]),
            Subset(index =  13,name = "XuYc", latex_name="X\\cup Y^c", regions=[0,2,3]),
            Subset(index =  14,name = "XcuY", latex_name="X^c\\cup Y", regions=[1,2,3]),  
            Subset(index =  15,name = "U", latex_name="U", regions=[0,1,2,3])
        ]

        for s in self.subsets:
            self.subsets_by_name[s.name] = s

class Venn3(Venn):
    def __init__(self):
        self.box_size = 4.2
        self.U_radius = 4
        self.points = []
        self.arcs = []
        self.regions = []
        self.subsets = []
        self.subsets_by_name = {}
        self.ax = None

        sigma = lambda n: ((n+1) % 3) + 3 * (n // 3)
        tau   = lambda n: ((n-1) % 3) + 3 * (n // 3)

        for i in range(3):
            x0 = np.cos(np.pi*(1/2 + 2*i/3))
            y0 = np.sin(np.pi*(1/2 + 2*i/3))
            p = Point(index = i, 
                      coords = (x0, y0), 
                      adjacent_regions = [i,tau(i+3),6,sigma(i+3)],
                      inward_arcs = [sigma(i+3),tau(i)],
                      outward_arcs = [sigma(i+6),tau(i+3)])
            self.points.append(p)

        for i in range(3):
            x0 = np.cos(np.pi*(1/2 + 2*i/3))
            y0 = np.sin(np.pi*(1/2 + 2*i/3))
            q = Point(index = i+3, 
                      coords = (-2*x0, -2*y0), 
                      adjacent_regions = [i+3,sigma(i),7,tau(i)],
                      inward_arcs = [tau(i+6),sigma(i+9)],
                      outward_arcs = [tau(i+9),sigma(i)])
            self.points.append(q)

        arcs = {}
        for i in range(3):
            x0 = np.cos(np.pi*(1/2 + 2*i/3))
            y0 = np.sin(np.pi*(1/2 + 2*i/3))
            types = ['intermediate acw','inner','intermediate cw','outer']
            starts = [tau(i+3), sigma(i), tau(i), sigma(i+3)]
            ends = [sigma(i), tau(i), sigma(i+3), tau(i+3)]
            left_regions = [tau(i+3),i+3,sigma(i+3),i]
            right_regions = [sigma(i),6,tau(i),7]
            colours = ['red','green','blue','magenta']
            for j in range(4):
                m = np.pi*(2*i/3+j/3-1)
                w = np.pi*(1 if j == 3 else 1/3)
                a = Arc(index=i+3*j, 
                        type=types[j],
                        centre = (x0, y0), 
                        radius = np.sqrt(3), 
                        min_angle = m, 
                        max_angle = m+w,
                        start = starts[j],
                        end = ends[j],
                        left_region = left_regions[j],
                        right_region = right_regions[j],
                        colour = colours[j])
                arcs[i+3*j] = a

        self.arcs = [arcs[i] for i in range(12)]

        for i in range(3):
            x0, y0 = self.points[i].coords
            x0 *= 7/4
            y0 *= 7/4
            r = Region(index=i,
                       type="outer",
                       centre=(x0, y0),
                       adjacent_points=[i,sigma(i+3),tau(i+3)],
                       adjacent_oriented_arcs=[[i+9,1],[sigma(i+6),-1],[tau(i),-1]],
                       components=[i])
            self.regions.append(r)
                
        for i in range(3):
            x0, y0 = self.points[i].coords
            x0 *= -5/4
            y0 *= -5/4
            r = Region(index=i+3,
                       type="outer",
                       centre=(x0, y0),
                       adjacent_points=[i+3,tau(i),sigma(i)],
                       adjacent_oriented_arcs=[[sigma(i),1],[i+3,-1],[tau(i+6),1]],
                       components=[i])
            self.regions.append(r)

        r = Region(index=6,
                   type="centre",
                   centre=(0,0),
                   adjacent_points=[0,1,2],
                   adjacent_oriented_arcs=[[3,1],[4,1],[5,1]],
                   components=[0,1,2])
        self.regions.append(r)

        r = Region(index=7,
                   type="exterior",
                   centre=(0,-11/4),
                   adjacent_points=[3,4,5],
                   adjacent_oriented_arcs=[[10,-1],[9,-1],[11,-1]],
                   components=[])
        self.regions.append(r)
        
        for i in range(8):
            self.regions[i].set_xys(self.arcs)

        r = self.regions[7]
        ts = np.linspace(-np.pi/2, 3*np.pi/2, 200)
        xs = np.cos(ts) * self.U_radius
        ys = np.sin(ts) * self.U_radius
        r.xs = np.array(list(r.xs) + list(xs) + [r.xs[0]])
        r.ys = np.array(list(r.ys) + list(ys) + [r.ys[0]])

        self.subsets = [
            Subset(index =   0,name = "O", latex_name="\\emptyset", regions=[]),
            Subset(index =   1,name = "X\\(YuZ)", latex_name="X\\setminus (Y\\cup Z)", regions=[0]),
            Subset(index =   2,name = "Y\\(XuZ)", latex_name="Y\\setminus (X\\cup Z)", regions=[1]),
            Subset(index =   3,name = "Z\\(XuY)", latex_name="Z\\setminus (X\\cup Y)", regions=[2]),
            Subset(index =   4,name = "(YnZ)\\X", latex_name="(Y\\cap Z)\\setminus X", regions=[3]),
            Subset(index =   5,name = "(XnZ)\\Y", latex_name="(X\\cap Z)\\setminus Y", regions=[4]),
            Subset(index =   6,name = "(XnY)\\Z", latex_name="(X\\cap Y)\\setminus Z", regions=[5]),
            Subset(index =   7,name = "XnYnZ", latex_name="X\\cap Y\\cap Z", regions=[6]),
            Subset(index =   8,name = "(XuYuZ)c", latex_name="(X\\cup Y\\cup Z)^c", regions=[7]),
            Subset(index =   9,name = "(XdY)\\Z", latex_name="(X\\Delta Y)\\setminus Z", regions=[0, 1]),
            Subset(index =  10,name = "(XdZ)\\Y", latex_name="(X\\Delta Z)\\setminus Y", regions=[0, 2]),
            Subset(index =  11,name = "03", latex_name="03", regions=[0, 3]),
            Subset(index =  12,name = "X\\Y", latex_name="X\\setminus Y", regions=[0, 4]),
            Subset(index =  13,name = "X\\Z", latex_name="X\\setminus Z", regions=[0, 5]),
            Subset(index =  14,name = "X\\(YdZ)", latex_name="X\\setminus (Y\\Delta Z)", regions=[0, 6]),
            Subset(index =  15,name = "(YuZ)c", latex_name="(Y\\cup Z)^c", regions=[0, 7]),
            Subset(index =  16,name = "(YdZ)\\X", latex_name="(Y\\Delta Z)\\setminus X", regions=[1, 2]),
            Subset(index =  17,name = "Y\\X", latex_name="Y\\setminus X", regions=[1, 3]),
            Subset(index =  18,name = "14", latex_name="14", regions=[1, 4]),
            Subset(index =  19,name = "Y\\Z", latex_name="Y\\setminus Z", regions=[1, 5]),
            Subset(index =  20,name = "Y\\(XdZ)", latex_name="Y\\setminus (X\\Delta Z)", regions=[1, 6]),
            Subset(index =  21,name = "(XuZ)c", latex_name="(X\\cup Z)^c", regions=[1, 7]),
            Subset(index =  22,name = "Z\\X", latex_name="Z\\setminus X", regions=[2, 3]),
            Subset(index =  23,name = "Z\\Y", latex_name="Z\\setminus Y", regions=[2, 4]),
            Subset(index =  24,name = "25", latex_name="25", regions=[2, 5]),
            Subset(index =  25,name = "Z\\(XdY)", latex_name="Z\\setminus (X\\Delta Y)", regions=[2, 6]),
            Subset(index =  26,name = "(XuY)c", latex_name="(X\\cup Y)^c", regions=[2, 7]),
            Subset(index =  27,name = "Zn(XdY)", latex_name="Z\\cap (X\\Delta Y)", regions=[3, 4]),
            Subset(index =  28,name = "Yn(XdZ)", latex_name="Y\\cap (X\\Delta Z)", regions=[3, 5]),
            Subset(index =  29,name = "YnZ", latex_name="Y\\cap Z", regions=[3, 6]),
            Subset(index =  30,name = "37", latex_name="37", regions=[3, 7]),
            Subset(index =  31,name = "Xn(YdZ)", latex_name="X\\cap (Y\\Delta Z)", regions=[4, 5]),
            Subset(index =  32,name = "XnZ", latex_name="X\\cap Z", regions=[4, 6]),
            Subset(index =  33,name = "47", latex_name="47", regions=[4, 7]),
            Subset(index =  34,name = "XnY", latex_name="X\\cap Y", regions=[5, 6]),
            Subset(index =  35,name = "57", latex_name="57", regions=[5, 7]),
            Subset(index =  36,name = "67", latex_name="67", regions=[6, 7]),
            Subset(index =  37,name = "012", latex_name="012", regions=[0, 1, 2]),
            Subset(index =  38,name = "013", latex_name="013", regions=[0, 1, 3]),
            Subset(index =  39,name = "014", latex_name="014", regions=[0, 1, 4]),
            Subset(index =  40,name = "(XuY)\\Z", latex_name="(X\\cup Y)\\setminus Z", regions=[0, 1, 5]),
            Subset(index =  41,name = "016", latex_name="016", regions=[0, 1, 6]),
            Subset(index =  42,name = "017", latex_name="017", regions=[0, 1, 7]),
            Subset(index =  43,name = "023", latex_name="023", regions=[0, 2, 3]),
            Subset(index =  44,name = "(XuZ)\\Y", latex_name="(X\\cup Z)\\setminus Y", regions=[0, 2, 4]),
            Subset(index =  45,name = "025", latex_name="025", regions=[0, 2, 5]),
            Subset(index =  46,name = "026", latex_name="026", regions=[0, 2, 6]),
            Subset(index =  47,name = "027", latex_name="027", regions=[0, 2, 7]),
            Subset(index =  48,name = "034", latex_name="034", regions=[0, 3, 4]),
            Subset(index =  49,name = "035", latex_name="035", regions=[0, 3, 5]),
            Subset(index =  50,name = "036", latex_name="036", regions=[0, 3, 6]),
            Subset(index =  51,name = "037", latex_name="037", regions=[0, 3, 7]),
            Subset(index =  52,name = "X\\(YnZ)", latex_name="X\\setminus (Y\\cap Z)", regions=[0, 4, 5]),
            Subset(index =  53,name = "046", latex_name="046", regions=[0, 4, 6]),
            Subset(index =  54,name = "047", latex_name="047", regions=[0, 4, 7]),
            Subset(index =  55,name = "056", latex_name="056", regions=[0, 5, 6]),
            Subset(index =  56,name = "057", latex_name="057", regions=[0, 5, 7]),
            Subset(index =  57,name = "067", latex_name="067", regions=[0, 6, 7]),
            Subset(index =  58,name = "(YuZ)\\X", latex_name="(Y\\cup Z)\\setminus X", regions=[1, 2, 3]),
            Subset(index =  59,name = "124", latex_name="124", regions=[1, 2, 4]),
            Subset(index =  60,name = "125", latex_name="125", regions=[1, 2, 5]),
            Subset(index =  61,name = "126", latex_name="126", regions=[1, 2, 6]),
            Subset(index =  62,name = "127", latex_name="127", regions=[1, 2, 7]),
            Subset(index =  63,name = "134", latex_name="134", regions=[1, 3, 4]),
            Subset(index =  64,name = "Y\\(XnZ)", latex_name="Y\\setminus (X\\cap Z)", regions=[1, 3, 5]),
            Subset(index =  65,name = "136", latex_name="136", regions=[1, 3, 6]),
            Subset(index =  66,name = "137", latex_name="137", regions=[1, 3, 7]),
            Subset(index =  67,name = "145", latex_name="145", regions=[1, 4, 5]),
            Subset(index =  68,name = "146", latex_name="146", regions=[1, 4, 6]),
            Subset(index =  69,name = "147", latex_name="147", regions=[1, 4, 7]),
            Subset(index =  70,name = "156", latex_name="156", regions=[1, 5, 6]),
            Subset(index =  71,name = "157", latex_name="157", regions=[1, 5, 7]),
            Subset(index =  72,name = "167", latex_name="167", regions=[1, 6, 7]),
            Subset(index =  73,name = "Z\\(XnY)", latex_name="Z\\setminus (X\\cap Y)", regions=[2, 3, 4]),
            Subset(index =  74,name = "235", latex_name="235", regions=[2, 3, 5]),
            Subset(index =  75,name = "236", latex_name="236", regions=[2, 3, 6]),
            Subset(index =  76,name = "237", latex_name="237", regions=[2, 3, 7]),
            Subset(index =  77,name = "245", latex_name="245", regions=[2, 4, 5]),
            Subset(index =  78,name = "246", latex_name="246", regions=[2, 4, 6]),
            Subset(index =  79,name = "247", latex_name="247", regions=[2, 4, 7]),
            Subset(index =  80,name = "256", latex_name="256", regions=[2, 5, 6]),
            Subset(index =  81,name = "257", latex_name="257", regions=[2, 5, 7]),
            Subset(index =  82,name = "267", latex_name="267", regions=[2, 6, 7]),
            Subset(index =  83,name = "345", latex_name="345", regions=[3, 4, 5]),
            Subset(index =  84,name = "Zn(XuY)", latex_name="Z\\cap (X\\cup Y)", regions=[3, 4, 6]),
            Subset(index =  85,name = "347", latex_name="347", regions=[3, 4, 7]),
            Subset(index =  86,name = "Yn(XuZ)", latex_name="Y\\cap (X\\cup Z)", regions=[3, 5, 6]),
            Subset(index =  87,name = "357", latex_name="357", regions=[3, 5, 7]),
            Subset(index =  88,name = "367", latex_name="367", regions=[3, 6, 7]),
            Subset(index =  89,name = "Xn(YuZ)", latex_name="X\\cap (Y\\cup Z)", regions=[4, 5, 6]),
            Subset(index =  90,name = "457", latex_name="457", regions=[4, 5, 7]),
            Subset(index =  91,name = "467", latex_name="467", regions=[4, 6, 7]),
            Subset(index =  92,name = "567", latex_name="567", regions=[5, 6, 7]),
            Subset(index =  93,name = "Xd(YuZ)", latex_name="X\\Delta (Y\\cup Z)", regions=[0, 1, 2, 3]),
            Subset(index =  94,name = "Yd(XuZ)", latex_name="Y\\Delta (X\\cup Z)", regions=[0, 1, 2, 4]),
            Subset(index =  95,name = "Zd(XuY)", latex_name="Z\\Delta (X\\cup Y)", regions=[0, 1, 2, 5]),
            Subset(index =  96,name = "XdYdZ", latex_name="X\\Delta Y\\Delta Z", regions=[0, 1, 2, 6]),
            Subset(index =  97,name = "0127", latex_name="0127", regions=[0, 1, 2, 7]),
            Subset(index =  98,name = "XdY", latex_name="X\\Delta Y", regions=[0, 1, 3, 4]),
            Subset(index =  99,name = "0135", latex_name="0135", regions=[0, 1, 3, 5]),
            Subset(index = 100,name = "0136", latex_name="0136", regions=[0, 1, 3, 6]),
            Subset(index = 101,name = "0137", latex_name="0137", regions=[0, 1, 3, 7]),
            Subset(index = 102,name = "0145", latex_name="0145", regions=[0, 1, 4, 5]),
            Subset(index = 103,name = "0146", latex_name="0146", regions=[0, 1, 4, 6]),
            Subset(index = 104,name = "0147", latex_name="0147", regions=[0, 1, 4, 7]),
            Subset(index = 105,name = "0156", latex_name="0156", regions=[0, 1, 5, 6]),
            Subset(index = 106,name = "Zc", latex_name="Z^c", regions=[0, 1, 5, 7]),
            Subset(index = 107,name = "0167", latex_name="0167", regions=[0, 1, 6, 7]),
            Subset(index = 108,name = "0234", latex_name="0234", regions=[0, 2, 3, 4]),
            Subset(index = 109,name = "XdZ", latex_name="X\\Delta Z", regions=[0, 2, 3, 5]),
            Subset(index = 110,name = "0236", latex_name="0236", regions=[0, 2, 3, 6]),
            Subset(index = 111,name = "0237", latex_name="0237", regions=[0, 2, 3, 7]),
            Subset(index = 112,name = "0245", latex_name="0245", regions=[0, 2, 4, 5]),
            Subset(index = 113,name = "0246", latex_name="0246", regions=[0, 2, 4, 6]),
            Subset(index = 114,name = "Yc", latex_name="Y^c", regions=[0, 2, 4, 7]),
            Subset(index = 115,name = "0256", latex_name="0256", regions=[0, 2, 5, 6]),
            Subset(index = 116,name = "0257", latex_name="0257", regions=[0, 2, 5, 7]),
            Subset(index = 117,name = "0267", latex_name="0267", regions=[0, 2, 6, 7]),
            Subset(index = 118,name = "Xd(YnZ)", latex_name="X\\Delta (Y\\cap Z)", regions=[0, 3, 4, 5]),
            Subset(index = 119,name = "0346", latex_name="0346", regions=[0, 3, 4, 6]),
            Subset(index = 120,name = "0347", latex_name="0347", regions=[0, 3, 4, 7]),
            Subset(index = 121,name = "0356", latex_name="0356", regions=[0, 3, 5, 6]),
            Subset(index = 122,name = "0357", latex_name="0357", regions=[0, 3, 5, 7]),
            Subset(index = 123,name = "(YdZ)c", latex_name="(Y\\Delta Z)^c", regions=[0, 3, 6, 7]),
            Subset(index = 124,name = "X", latex_name="X", regions=[0, 4, 5, 6]),
            Subset(index = 125,name = "0457", latex_name="0457", regions=[0, 4, 5, 7]),
            Subset(index = 126,name = "0467", latex_name="0467", regions=[0, 4, 6, 7]),
            Subset(index = 127,name = "0567", latex_name="0567", regions=[0, 5, 6, 7]),
            Subset(index = 128,name = "1234", latex_name="1234", regions=[1, 2, 3, 4]),
            Subset(index = 129,name = "1235", latex_name="1235", regions=[1, 2, 3, 5]),
            Subset(index = 130,name = "1236", latex_name="1236", regions=[1, 2, 3, 6]),
            Subset(index = 131,name = "Xc", latex_name="X^c", regions=[1, 2, 3, 7]),
            Subset(index = 132,name = "YdZ", latex_name="Y\\Delta Z", regions=[1, 2, 4, 5]),
            Subset(index = 133,name = "1246", latex_name="1246", regions=[1, 2, 4, 6]),
            Subset(index = 134,name = "1247", latex_name="1247", regions=[1, 2, 4, 7]),
            Subset(index = 135,name = "1345", latex_name="1345", regions=[1, 2, 5, 6]),
            Subset(index = 136,name = "1257", latex_name="1257", regions=[1, 2, 5, 7]),
            Subset(index = 137,name = "1267", latex_name="1267", regions=[1, 2, 6, 7]),
            Subset(index = 138,name = "Yd(XnZ)", latex_name="Y\\Delta (X\\cap Z)", regions=[1, 3, 4, 5]),
            Subset(index = 139,name = "1346", latex_name="1346", regions=[1, 3, 4, 6]),
            Subset(index = 140,name = "1347", latex_name="1347", regions=[1, 3, 4, 7]),
            Subset(index = 141,name = "Y", latex_name="Y", regions=[1, 3, 5, 6]),
            Subset(index = 142,name = "1357", latex_name="1357", regions=[1, 3, 5, 7]),
            Subset(index = 143,name = "1367", latex_name="1367", regions=[1, 3, 6, 7]),
            Subset(index = 144,name = "1456", latex_name="1456", regions=[1, 4, 5, 6]),
            Subset(index = 145,name = "1457", latex_name="1457", regions=[1, 4, 5, 7]),
            Subset(index = 146,name = "(XdZ)c", latex_name="(X\\Delta Z)^c", regions=[1, 4, 6, 7]),
            Subset(index = 147,name = "1567", latex_name="1567", regions=[1, 5, 6, 7]),
            Subset(index = 148,name = "Zd(XnY)", latex_name="Z\\Delta (X\\cap Y)", regions=[2, 3, 4, 5]),
            Subset(index = 149,name = "Z", latex_name="Z", regions=[2, 3, 4, 6]),
            Subset(index = 150,name = "2347", latex_name="2347", regions=[2, 3, 4, 7]),
            Subset(index = 151,name = "2356", latex_name="2356", regions=[2, 3, 5, 6]),
            Subset(index = 152,name = "2357", latex_name="2357", regions=[2, 3, 5, 7]),
            Subset(index = 153,name = "2367", latex_name="2367", regions=[2, 3, 6, 7]),
            Subset(index = 154,name = "2456", latex_name="2456", regions=[2, 4, 5, 6]),
            Subset(index = 155,name = "2457", latex_name="2457", regions=[2, 4, 5, 7]),
            Subset(index = 156,name = "2467", latex_name="2467", regions=[2, 4, 6, 7]),
            Subset(index = 157,name = "(XdY)c", latex_name="(X\\Delta Y)^c", regions=[2, 5, 6, 7]),
            Subset(index = 158,name = "3456", latex_name="3456", regions=[3, 4, 5, 6]),
            Subset(index = 159,name = "3457", latex_name="3457", regions=[3, 4, 5, 7]),
            Subset(index = 160,name = "3467", latex_name="3467", regions=[3, 4, 6, 7]),
            Subset(index = 161,name = "3567", latex_name="3567", regions=[3, 5, 6, 7]),
            Subset(index = 162,name = "4567", latex_name="4567", regions=[4, 5, 6, 7]),
            Subset(index = 163,name = "01234", latex_name="01234", regions=[0, 1, 2, 3, 4]),
            Subset(index = 164,name = "01235", latex_name="01235", regions=[0, 1, 2, 3, 5]),
            Subset(index = 165,name = "01236", latex_name="01236", regions=[0, 1, 2, 3, 6]),
            Subset(index = 166,name = "01237", latex_name="01237", regions=[0, 1, 2, 3, 7]),
            Subset(index = 167,name = "01245", latex_name="01245", regions=[0, 1, 2, 4, 5]),
            Subset(index = 168,name = "01246", latex_name="01246", regions=[0, 1, 2, 4, 6]),
            Subset(index = 169,name = "01247", latex_name="01247", regions=[0, 1, 2, 4, 7]),
            Subset(index = 170,name = "01256", latex_name="01256", regions=[0, 1, 2, 5, 6]),
            Subset(index = 171,name = "01257", latex_name="01257", regions=[0, 1, 2, 5, 7]),
            Subset(index = 172,name = "01267", latex_name="01267", regions=[0, 1, 2, 6, 7]),
            Subset(index = 173,name = "01345", latex_name="01345", regions=[0, 1, 3, 4, 5]),
            Subset(index = 174,name = "01346", latex_name="01346", regions=[0, 1, 3, 4, 6]),
            Subset(index = 175,name = "01347", latex_name="01347", regions=[0, 1, 3, 4, 7]),
            Subset(index = 176,name = "01356", latex_name="01356", regions=[0, 1, 3, 5, 6]),
            Subset(index = 177,name = "01357", latex_name="01357", regions=[0, 1, 3, 5, 7]),
            Subset(index = 178,name = "01367", latex_name="01367", regions=[0, 1, 3, 6, 7]),
            Subset(index = 179,name = "01456", latex_name="01456", regions=[0, 1, 4, 5, 6]),
            Subset(index = 180,name = "01457", latex_name="01457", regions=[0, 1, 4, 5, 7]),
            Subset(index = 181,name = "01467", latex_name="01467", regions=[0, 1, 4, 6, 7]),
            Subset(index = 182,name = "01567", latex_name="01567", regions=[0, 1, 5, 6, 7]),
            Subset(index = 183,name = "02345", latex_name="02345", regions=[0, 2, 3, 4, 5]),
            Subset(index = 184,name = "02346", latex_name="02346", regions=[0, 2, 3, 4, 6]),
            Subset(index = 185,name = "02347", latex_name="02347", regions=[0, 2, 3, 4, 7]),
            Subset(index = 186,name = "02356", latex_name="02356", regions=[0, 2, 3, 5, 6]),
            Subset(index = 187,name = "02357", latex_name="02357", regions=[0, 2, 3, 5, 7]),
            Subset(index = 188,name = "02367", latex_name="02367", regions=[0, 2, 3, 6, 7]),
            Subset(index = 189,name = "02456", latex_name="02456", regions=[0, 2, 4, 5, 6]),
            Subset(index = 190,name = "02457", latex_name="02457", regions=[0, 2, 4, 5, 7]),
            Subset(index = 191,name = "02467", latex_name="02467", regions=[0, 2, 4, 6, 7]),
            Subset(index = 192,name = "02567", latex_name="02567", regions=[0, 2, 5, 6, 7]),
            Subset(index = 193,name = "Xu(YnZ)", latex_name="X\\cup (Y\\cap Z)", regions=[0, 3, 4, 5, 6]),
            Subset(index = 194,name = "03457", latex_name="03457", regions=[0, 3, 4, 5, 7]),
            Subset(index = 195,name = "03467", latex_name="03467", regions=[0, 3, 4, 6, 7]),
            Subset(index = 196,name = "03567", latex_name="03567", regions=[0, 3, 5, 6, 7]),
            Subset(index = 197,name = "04567", latex_name="04567", regions=[0, 4, 5, 6, 7]),
            Subset(index = 198,name = "12345", latex_name="12345", regions=[1, 2, 3, 4, 5]),
            Subset(index = 199,name = "12346", latex_name="12346", regions=[1, 2, 3, 4, 6]),
            Subset(index = 200,name = "12347", latex_name="12347", regions=[1, 2, 3, 4, 7]),
            Subset(index = 201,name = "12356", latex_name="12356", regions=[1, 2, 3, 5, 6]),
            Subset(index = 202,name = "12357", latex_name="12357", regions=[1, 2, 3, 5, 7]),
            Subset(index = 203,name = "12367", latex_name="12367", regions=[1, 2, 3, 6, 7]),
            Subset(index = 204,name = "12456", latex_name="12456", regions=[1, 2, 4, 5, 6]),
            Subset(index = 205,name = "12457", latex_name="12457", regions=[1, 2, 4, 5, 7]),
            Subset(index = 206,name = "12467", latex_name="12467", regions=[1, 2, 4, 6, 7]),
            Subset(index = 207,name = "12567", latex_name="12567", regions=[1, 2, 5, 6, 7]),
            Subset(index = 208,name = "Yu(XnZ)", latex_name="Y\\cup (X\\cap Z)", regions=[1, 3, 4, 5, 6]),
            Subset(index = 209,name = "13457", latex_name="13457", regions=[1, 3, 4, 5, 7]),
            Subset(index = 210,name = "13467", latex_name="13467", regions=[1, 3, 4, 6, 7]),
            Subset(index = 211,name = "13567", latex_name="13567", regions=[1, 3, 5, 6, 7]),
            Subset(index = 212,name = "14567", latex_name="14567", regions=[1, 4, 5, 6, 7]),
            Subset(index = 213,name = "Zu(XnY)", latex_name="Z\\cup (X\\cap Y)", regions=[2, 3, 4, 5, 6]),
            Subset(index = 214,name = "23457", latex_name="23457", regions=[2, 3, 4, 5, 7]),
            Subset(index = 215,name = "23467", latex_name="23467", regions=[2, 3, 4, 6, 7]),
            Subset(index = 216,name = "23567", latex_name="23567", regions=[2, 3, 5, 6, 7]),
            Subset(index = 217,name = "24567", latex_name="24567", regions=[2, 4, 5, 6, 7]),
            Subset(index = 218,name = "34567", latex_name="34567", regions=[3, 4, 5, 6, 7]),
            Subset(index = 219,name = "012345", latex_name="012345", regions=[0, 1, 2, 3, 4, 5]),
            Subset(index = 220,name = "Zu(XdY)", latex_name="Z\\cup (X\\Delta Y)", regions=[0, 1, 2, 3, 4, 6]),
            Subset(index = 221,name = "(XnY)c", latex_name="(X\\cap Y)^c", regions=[0, 1, 2, 3, 4, 7]),
            Subset(index = 222,name = "Yu(XdZ)", latex_name="Y\\cup (X\\Delta Z)", regions=[0, 1, 2, 3, 5, 6]),
            Subset(index = 223,name = "(XnZ)c", latex_name="(X\\cap Z)^c", regions=[0, 1, 2, 3, 5, 7]),
            Subset(index = 224,name = "012367", latex_name="012367", regions=[0, 1, 2, 3, 6, 7]),
            Subset(index = 225,name = "Xu(YdZ)", latex_name="X\\cup (Y\\Delta Z)", regions=[0, 1, 2, 4, 5, 6]),
            Subset(index = 226,name = "(YnZ)c", latex_name="(Y\\cap Z)^c", regions=[0, 1, 2, 4, 5, 7]),
            Subset(index = 227,name = "012467", latex_name="012467", regions=[0, 1, 2, 4, 6, 7]),
            Subset(index = 228,name = "012567", latex_name="012567", regions=[0, 1, 2, 5, 6, 7]),
            Subset(index = 229,name = "XuY", latex_name="X\\cup Y", regions=[0, 1, 3, 4, 5, 6]),
            Subset(index = 230,name = "013457", latex_name="013457", regions=[0, 1, 3, 4, 5, 7]),
            Subset(index = 231,name = "013467", latex_name="013467", regions=[0, 1, 3, 4, 6, 7]),
            Subset(index = 232,name = "ZcuY", latex_name="Z^c\\cup Y", regions=[0, 1, 3, 5, 6, 7]),
            Subset(index = 233,name = "ZcuX", latex_name="Z^c\\cup X", regions=[0, 1, 4, 5, 6, 7]),
            Subset(index = 234,name = "XuZ", latex_name="X\\cup Z", regions=[0, 2, 3, 4, 5, 6]),
            Subset(index = 235,name = "023457", latex_name="023457", regions=[0, 2, 3, 4, 5, 7]),
            Subset(index = 236,name = "YcuZ", latex_name="Y^c\\cup Z", regions=[0, 2, 3, 4, 6, 7]),
            Subset(index = 237,name = "023567", latex_name="023567", regions=[0, 2, 3, 5, 6, 7]),
            Subset(index = 238,name = "YcuX", latex_name="Y^c\\cup X", regions=[0, 2, 4, 5, 6, 7]),
            Subset(index = 239,name = "034567", latex_name="034567", regions=[0, 3, 4, 5, 6, 7]),
            Subset(index = 240,name = "YuZ", latex_name="Y\\cup Z", regions=[1, 2, 3, 4, 5, 6]),
            Subset(index = 241,name = "123457", latex_name="123457", regions=[1, 2, 3, 4, 5, 7]),
            Subset(index = 242,name = "XcuZ", latex_name="X^c\\cup Z", regions=[1, 2, 3, 4, 6, 7]),
            Subset(index = 243,name = "XcuY", latex_name="X^c\\cup Y", regions=[1, 2, 3, 5, 6, 7]),
            Subset(index = 244,name = "124567", latex_name="124567", regions=[1, 2, 4, 5, 6, 7]),
            Subset(index = 245,name = "134567", latex_name="134567", regions=[1, 3, 4, 5, 6, 7]),
            Subset(index = 246,name = "234567", latex_name="234567", regions=[2, 3, 4, 5, 6, 7]),
            Subset(index = 247,name = "XuYuZ", latex_name="X\\cup Y\\cup Z", regions=[0, 1, 2, 3, 4, 5, 6]),
            Subset(index = 248,name = "(XnYnZ)c", latex_name="(X\\cap Y\\cap Z)^c", regions=[0, 1, 2, 3, 4, 5, 7]),
            Subset(index = 249,name = "XcuYcuZ", latex_name="X^c\\cup Y^c\\cup Z", regions=[0, 1, 2, 3, 4, 6, 7]),
            Subset(index = 250,name = "XcuYuZc", latex_name="X^c\\cup Y\\cup Z^c", regions=[0, 1, 2, 3, 5, 6, 7]),
            Subset(index = 251,name = "XuYcuZc", latex_name="X\\cup Y^c\\cup Z^c", regions=[0, 1, 2, 4, 5, 6, 7]),
            Subset(index = 252,name = "XuYuZc", latex_name="X\\cup Y\\cup Z^c", regions=[0, 1, 3, 4, 5, 6, 7]),
            Subset(index = 253,name = "XuYcuZ", latex_name="X\\cup Y^c\\cup Z", regions=[0, 2, 3, 4, 5, 6, 7]),
            Subset(index = 254,name = "XcuYuZ", latex_name="X^c\\cup Y\\cup Z", regions=[1, 2, 3, 4, 5, 6, 7]),
            Subset(index = 255,name = "U", latex_name="U", regions=[0, 1, 2, 3, 4, 5, 6, 7])
        ]

        for s in self.subsets:
            self.subsets_by_name[s.name] = s


