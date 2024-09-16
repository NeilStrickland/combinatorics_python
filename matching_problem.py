import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from IPython.display import display, Math, Latex, HTML

class Tree:
    def __init__(self,root,children=None):
        self.root = root
        self.children = [] if children is None else children

    def show(self, prefix='', indent=' '):
        print(prefix + str(self.root))
        for c in self.children:
            c.show(prefix + indent, indent)

    def list(self, prefix = None):
        if prefix is None:
            prefix = []
        a = self.root
        prefix1 = prefix if a is None else prefix + [a]
        L = [prefix1]
        for x in self.children:
            L.extend(x.list(prefix1))
        return L
    
class MatchingProblem:
    """
    Represents a matching problem, consisting of sets A and B, and a subset E of A x B.

    Attributes:
    left_set: the set A (represented as a list)

    right_set: the set B (represented as a list)

    left_order: the number of elements in A

    right_order: the number of elements in B

    product_set: the set A x B

    relation: the subset E of A x B

    coblock: a dictionary mapping elements of A to their coblocks, so 
        coblock[a] = {b in B : (a,b) in E}
        The coblocks are also called row sets, and denoted by R[a]

    block: a dictionary mapping elements of B to their coblocks, so
        block[b] = {a in A : (a,b) in E}
        The blocks are also called column sets or candidate sets, 
        and denoted by C[b]

    left_index: a dictionary mapping elements of A to their indices

    right_index: a dictionary mapping elements of B to their indices

    index: a dictionary mapping pairs (a,b) to their index pairs

    indexed_relation: the set of index pairs corresponding to E

    relation_matrix: a binary matrix M representing of E, in the sense 
        that M[i,j] = 1 if (A[i],B[j]) in E

    indexed_rook_tree: a tree representing the possible partial matchings
        The top node has root=None, and every other node has a root (i,j)
        with (A[i],B[j]) in E.  The children of a node (i,j) are the nodes
        (k,l) such that (a) (i,j) < (k,l) in the lexicographic order, and
        (b) The set consisting of (k,l), (i,j) and all ancestors is a 
        partial matching.

    indexed_rook_list: a list of all possible partial matchings, represented
        as lists of index pairs.

    rook_polynomial: The rook polynomial of the matching problem, expressed
        as a numpy array of coefficients.  The length of the array is 
        min(|A|,|B|) + 1, and the coefficient at index i is the number of
        partial matchings of size i.

    indexed_matchings: a list of all matchings, represented as lists of index
        pairs.  A matching is a partial matching of size |A|.

    indexed_comatchings: a list of all comatchings, represented as lists of index
        pairs.  A comatching is a partial matching of size |B|.

    rook_list: a list of all possible partial matchings, represented as lists
        of elements of A x B (so there is a bijection with indexed_rook_list).

    matchings: a list of all matchings, represented as lists of elements of A x B
        (so there is a bijection with indexed_matchings).

    comatchings: a list of all comatchings, represented as lists of elements of A x B
        (so there is a bijection with indexed_comatchings).

    num_matchings: the number of matchings

    num_comatchings: the number of comatchings

    maximal_matching_size: the size of the largest matching

    maximal_matchings: a list of all maximal matchings

    right_power_set: the set of subsets of B

    plausible_right_sets: a list of all plausible subsets T of B, where T is plausible if
        the candidate set C[T] = {a in A : there exists b in T with (a,b) in E}
        has |R[T]| >= |T|.

    implausible_right_sets: a list of all implausible subsets T of B, for which |C[T]| < |T|.

    tricky_right_sets: a list of all tricky subsets T of B, for which |C[T]| = |T|.
        (The notes use the phrase "barely plausible" for these sets.)

    left_power_set: the set of subsets of A

    plausible_left_sets: a list of all plausible subsets S of A, where S is plausible if
        the row set R[S] = {b in B : there exists a in S with (a,b) in E}
        has |R[S]| >= |S|.  (The notes mostly consider plausible right rather than 
        plausible left sets rather than plausible left sets.)

    implausible_left_sets: a list of all implausible subsets S of A, for which |R[S]| < |S|.

    tricky_left_sets: a list of all tricky subsets S of A, for which |R[S]| = |S|.
        (The notes use the phrase "barely plausible" for these sets.)

    """
    def __init__(self,A,B,E,fast=False):
        """
        Initializes a matching problem with sets A and B, and subset E of A x B.

        If fast=False, a full analysis of the matching problem is performed, including
        the computation of the rook tree, rook list, rook polynomial, matchings, comatchings,
        maximal matchings, plausible sets, implausible sets, tricky sets, plausible cosets,
        implausible cosets, and tricky cosets.  This will be slow unless A and B are small.

        If fast=True, only the basic attributes are computed.
        """
        if isinstance(A,int):
            A = list(range(A))
        else:
            A = sorted(list(A))
        if not(isinstance(A,list)):
            raise ValueError("A must be convertible to a list")
        nA = len(A)

        if isinstance(B,int):
            B = list(range(B))
        else:
            B = sorted(list(B))
        if not(isinstance(B,list)):
            raise ValueError("B must be convertible to a list")
        nB = len(B)

        if isinstance(E,list) or isinstance(E,set):
            E = sorted(list(E))
        else:
            raise ValueError("E must be a list or set")
        
        AB = [(a,b) for a in A for b in B]

        if not(all([x in AB for x in E])):
            raise ValueError("E must be a subset of A x B")
        
        self.left_set = A
        self.right_set = B
        self.left_order = nA
        self.right_order = nB
        self.product_set = AB
        self.relation = E
        self.block = {}
        self.coblock = {}
        self.left_index = {}
        self.right_index = {}
        self.index = {}

        for i in range(nA):
            self.left_index[A[i]] = i
        for i in range(nB):
            self.right_index[B[i]] = i

        for i in range(nA):
            for j in range(nB):
                self.index[(A[i],B[j])] = (i,j)
        
        for a in A:
            self.coblock[a] = {b for b in B if (a,b) in E}

        for b in B:
            self.block[b] = {a for a in A if (a,b) in E}

        self.indexed_relation = sorted([(self.left_index[a],self.right_index[b]) for a,b in E])

        M = np.zeros((nA,nB), dtype=int)
        for a,b in E:
            M[self.left_index[a],self.right_index[b]] = 1
        self.relation_matrix = M

        if not(fast):
            self.full_init()

    def full_init(self):
        A1 = list(range(self.left_order))
        B1 = list(range(self.right_order))
        T1 = MatchingProblem.rook_tree(A1,B1,self.indexed_relation)
        T2 = MatchingProblem.stepwise_rook_tree(A1,B1,self.indexed_relation)
        L1 = T1.list()
        L2 = T2.list()
        p1 = np.zeros(min(self.left_order,self.right_order)+1,dtype=int)
        for m in L1:
            p1[len(m)] += 1
        self.indexed_rook_tree = T1
        self.indexed_stepwise_rook_tree = T2
        self.indexed_rook_list = L1
        self.indexed_stepwise_rook_list = L2
        self.rook_polynomial = p1
        x = sp.symbols('x')
        self.rook_polynomial_latex = sp.latex(sum(p1[i]*x**i for i in range(len(p1))))
        self.indexed_matchings = [m for m in L1 if len(m) == self.left_order]
        self.indexed_comatchings = [m for m in L1 if len(m) == self.right_order]
        f = lambda v : (self.left_set[v[0]],self.right_set[v[1]])
        self.rook_list   = [list(map(f,u)) for u in self.indexed_rook_list]
        self.matchings   = [list(map(f,u)) for u in self.indexed_matchings]
        self.comatchings = [list(map(f,u)) for u in self.indexed_comatchings]
        f = lambda v : (self.left_set[v[0]],self.right_set[v[1]], v[2])
        self.stepwise_rook_list = [list(map(f,u)) for u in self.indexed_stepwise_rook_list]
        self.num_matchings = len(self.matchings)
        self.num_comatchings = len(self.comatchings)
        m = max(len(u) for u in self.rook_list)
        self.maximal_matching_size = m
        self.maximal_matchings = [u for u in self.rook_list if len(u) == m]
        self.rook_list_by_size = [[u for u in self.rook_list if len(u) == i] for i in range(m+1)]
        self.plausible_right_sets = []
        self.implausible_right_sets = []
        self.tricky_right_sets = []
        Q = [set()]
        for b in list(self.right_set):
            Q.extend([q.union({b}) for q in Q])
        self.right_power_set = Q
        for T in Q:
            S = set()
            for b in T:
                S = S.union(self.block[b])
            if len(S) < len(T):
                self.implausible_right_sets.append(S)
            else:
                self.plausible_right_sets.append(S)
                if len(S) == len(T):
                    self.tricky_right_sets.append(S)
        self.plausible_left_sets = []
        self.implausible_left_sets = []
        self.tricky_left_sets = []
        P = [set()]
        for a in list(self.left_set):
            P.extend([p.union({a}) for p in P])
        self.left_power_set = P
        for S in P:
            T = set()
            for a in S:
                T = T.union(self.coblock[a])
            if len(T) < len(S):
                self.implausible_left_sets.append(S)
            else:
                self.plausible_left_sets.append(S)
                if len(T) == len(S):
                    self.tricky_left_sets.append(S)
        
    def check_design(self):
        """
        Checks if the matching problem is a block design.  This means that there 
        are integers v,b,r,k, and lambda such that:
        (1) The left set has size v
        (2) The right set has size b
        (3) For each element of the left set, the corresponding coblock has size r
        (4) For each element of the right set, the corresponding block has size k
        (5) For any pair of distinct elements of the left set, the intersection 
            of their blocks has size lambda.

        In this context, it is conventional to call the left set V and the right set B,
        so the blocks are R[p] for p in V and the blocks are C[j] for j in B.  In
        this notation the conditions are |V|=v, |B|=b, |R[p]|=r, |C[j]|=k, and
        |R[p] intersect R[q]| = lambda for p != q.

        If the matching problem is a block design, the method returns True and sets
        the attributes design_v, design_b, design_r, design_k, and design_lambda to the
        appropriate values, and also sets the attribute design_params to the tuple
        (v,b,r,k,lambda).  If the matching problem is not a block design, the method
        returns False and sets the other design attributes to None.
        """
        self.design_v = self.left_order
        self.design_b = self.right_order
        kk = {len(self.block[j]) for j in self.right_set}
        if len(kk) == 1:
            self.design_k = kk.pop()
        else:
            self.design_k = None
        rr = {len(self.coblock[p]) for p in self.left_set}
        if len(rr) == 1:
            self.design_r = rr.pop()
        else:
            self.design_r = None
        ll = []
        for i1 in range(self.left_order):
            p1 = self.left_set[i1]
            for i0 in range(i1):
                p0 = self.left_set[i0]
                X = [j for j in self.coblock[p0] if j in self.coblock[p1]]
                ll.append(len(X))
        if len(set(ll)) == 1:
            self.design_lambda = ll.pop()
        else:
            self.design_lambda = None
        if self.design_k is None or self.design_r is None or self.design_lambda is None:
            self.is_design = False
            self.design_params = None
            self.design_report = "The matching problem is not a block design."
        else:
            self.is_design = True
            self.design_params = (self.design_v,self.design_b,self.design_r,self.design_k,self.design_lambda)
            self.design_report = "The matching problem is a block design with parameters " + \
                                 "$(v,b,r,k,\\lambda) = " + str(self.design_params) + "$"
        return self.is_design
    
    def check_plausibility(self, U=None):
        U0 = U
        if U is None:
            if len(self.implausible_right_sets) > 0:
                U = self.implausible_right_sets[0]
            else:
                U = []
        V = sorted(list({a for b in U for a in self.block[b]}))
        Us = '$\\{' + ', '.join(str(a) for a in U) + '\\}$'
        Vs = '$\\{' + ', '.join(str(a) for a in V) + '\\}$'
        if U0 is None:
            if len(U) > 0:
                msg = "The set " + Us + " (with candidate set " + Vs + ") is implausible," + \
                      "so the matching problem is implausible and cannot be solved."
                return (False,msg)
            else:
                return (True,"The matching problem is plausible, so Hall's Theorem guarantees that it has a solution.")
        else:
            if len(V) < len(U):
                msg = "The set " + Us + " (with candidate set " + Vs + ") is implausible. " + \
                      "As there is at least one implausible set, the matching problem is implausible " + \
                      "and cannot be solved."
                return (False,msg)
            elif len(V) == len(U):
                msg = "The set " + Us + " (with candidate set " + Vs + ") is barely plausible. " + \
                      "The matching problem may or may not be solvable; we still need to check " + \
                      "all the other subsets of $B$ for plausibility."
                return (True,msg)
            else:
                msg = "The set " + Us + " (with candidate set " + Vs + ") is very plausible. " + \
                      "The matching problem may or may not be solvable; we still need to check " + \
                      "all the other subsets of $B$ for plausibility."
                return (True,msg)
            
    def complement(self,fast = False):
        """
        Returns the complement of the matching problem, which has the same left set and right set,
        but the relation is the complement of the original relation.
        """
        Ec = sorted([ab for ab in self.product_set if ab not in self.relation])
        return MatchingProblem(self.left_set,self.right_set,Ec,fast)

    def restrict(self,A0,B0,fast=False):
        """
        Returns the restriction of the matching problem to the sets A0 and B0, where A0 is a subset
        of the left set and B0 is a subset of the right set.  The restriction has the same relation
        as the original matching problem, but only includes pairs (a,b) where a is in A0 and b is in B0.
        """
        E0 = [(a,b) for a,b in self.relation if a in A0 and b in B0]
        return MatchingProblem(A0,B0,E0,fast)

    def completion_problem(self,M,fast=False):
        """
        This assumes that M is a partial matching, represented as a list of index pairs
        [(a1,b1),...,(an,bn)].  It constructs a new matching problem P, such that 
        solving P is equivalent to finding a completion of M to a full matching. 
        In more detail, the left set of P is A0 = A \ {a1,...,an}, the right set of P is
        B0 = B \ {b1,...,bn}, and relation consists of the pairs (a,b) in E such that 
        a is in A0 and b is in B0.
        """
        A1 = [a for a,b in M]
        B1 = [b for a,b in M]
        A0 = [a for a in self.left_set if a not in A1]
        B0 = [b for b in self.right_set if b not in B1]
        E0 = [(a,b) for a,b in self.relation if a in A0 and b in B0]
        return MatchingProblem(A0,B0,E0,fast)

    def block_and_strip(self,a,b,fast=False):
        """
        Returns a pair of matching problems obtained from this one by blocking and stripping.
        It is assumed that (a,b) is an element of E.  Blocking means removing the pair (a,b)
        from E, and stripping means removing the elements a and b from the sets A and B, and
        removing the whole row containing a and the whole column containing b.
        """
        if not (a in self.left_set and b in self.right_set and (a,b) in self.relation):
            raise ValueError("(a,b) must be an element of E")
        A0 = self.left_set.copy()
        B0 = self.right_set.copy()
        E0 = [x for x in self.relation if x != (a,b)]
        A1 = [x for x in A0 if x != a]
        B1 = [x for x in B0 if x != b]
        E1 = [ab for ab in E0 if ab[0] != a and ab[1] != b]
        M0 = MatchingProblem(A0,B0,E0,fast)
        M1 = MatchingProblem(A1,B1,E1,fast)
        return (M0,M1)

    def factor(self,A0,fast=False):
        """
        This assumes that A0 is a subset of the left set A, with complement 
        A1 = A \ A0, and that B can be split into B0 and B1 such that the
        relation E is the union of E0 = E n (A0 x B0) and E1 = E n (A1 x B1). 
        If this is the case, the method returns a pair of matching problems
        obtained by restricting the original matching problem to (A0,B0) and
        to (A1,B1).
        """
        if not all([a in self.left_set for a in A0]):
            raise ValueError("A0 must be contained in the left set")
        A0 = A0.copy()
        A1 = [a for a in self.left_set if a not in A0]
        B0 = sorted(list({b for a in A0 for b in self.coblock[a]}))
        B1 = sorted(list({b for a in A1 for b in self.coblock[a]}))
        B01 = {b for b in B0 if b in B1}
        if len(B01) > 0:
            raise ValueError("B0 and B1 must be disjoint")
        E0 = [(a,b) for a,b in self.relation if a in A0 and b in B0]
        E1 = [(a,b) for a,b in self.relation if a in A1 and b in B1]
        M0 = MatchingProblem(A0,B0,E0,fast)
        M1 = MatchingProblem(A1,B1,E1,fast)
        return (M0,M1)
    
    def find_minimal_factors(self):
        """
        Finds all minimal nonempty subsets A0 of the left set A such that 
        the factor() method can be applied to A0.
        """
        F = []
        R = self.left_set.copy()
        while len(R) > 0:
            a = R[0]
            n0 = 0
            A0 = {a}
            while len(A0) > n0:
                n0 = len(A0)
                B0 = {b0 for a0 in A0 for b0 in self.coblock[a0]}
                A0 = {a0 for b0 in B0 for a0 in self.block[b0]}
            F.append(sorted(list(A0)))
            R = [a for a in R if a not in A0]
        return F
    
    def add_multiplicities(self,m,fast=False):
        """
        This creates a new matching problem by adding multiplicities, as in the process
        that constructs a team allocation problem from a job allocation problem.

        It is assumed that m is either a dictionary giving a nonnegative integer multiplicity
        for each element of the right set B, or a list of nonnegative integers of the 
        same length as B.  In the new matching problem, the right set consists of pairs
        (b,i) where b is in B and i is a nonnegative integer less than the multiplicity of b.
        The relation consists of pairs (a,(b,i)) where (a,b) is in the original relation E.
        """
        if isinstance(m,dict):
            if not all(isinstance(m[b],int) and m[b] >= 0 for b in self.right_set):
                raise ValueError("Multiplicities must be nonnegative integers")
            M = m
        elif isinstance(m,list):
            if not all(isinstance(m[i],int) and m[i] >= 0 for i in range(len(m))):
                raise ValueError("Multiplicities must be nonnegative integers")
            M = {b : m[i] for i,b in enumerate(self.right_set)}
        else:
            raise ValueError("Multiplicities must be a dictionary or list")
        B0 = [(b,i) for b in self.right_set for i in range(M[b])]
        E0 = [(a,(b,i)) for a,b in self.relation for i in range(M[b])]
        return MatchingProblem(self.left_set,B0,E0,fast)
    
    def classify_squares(self, matching=None):
        if matching is None:
            matching = []
        T = {}
        A = self.left_set
        B = self.right_set
        E = self.relation
 
        for a in A:
            for b in B:
                T[(a,b)] = "free" if (a,b) in E else "banned"

        for ab in matching:
            if T[ab] == "banned":
                T[ab] = "ban_broken"
            else:
                T[ab] = "occupied"

        for (a,b) in matching:
            for x in A: 
                if x != a:
                    if T[(x,b)] == "free":
                        T[(x,b)] = "blocked"
                    elif T[(x,b)] == "occupied":
                        T[(x,b)] = "block_broken"
            for y in B:
                if y != b:
                    if T[(a,y)] == "free":
                        T[(a,y)] = "blocked"
                    elif T[(a,y)] == "occupied":
                        T[(a,y)] = "block_broken"
        return T

    @staticmethod
    def draw_rook_raw(ax, x, y, scale, colour):
        rook_paths = [
            [[-13.5,13.5,13.5,-13.5,-13.5],[-15,-15,-12,-12,-15]],
            [[-10.5,-10.5,10.5,10.5,-10.5],[-12,-8,-8,-12,-12]],
            [[-11.5,-11.5,-7.5,-7.5,-2.5,-2.5,2.5,2.5,7.5,7.5,11.5,11.5,8.5,-8.5,-11.5],[10,15,15,13,13,15,15,13,13,15,15,10,7,7,10]],
            [[8.5,8.5,-8.5,-8.5],[7,-5.5,-5.5,7]],
            [[8.5,10,-10,-8.5],[-5.5,-8,-8,-5.5]],
            [[-11.5,11.5],[10,10]]]
        for path in rook_paths:
            ax.plot([scale*p/40+x for p in path[0]],[scale*p/40+y for p in path[1]],color=colour)

    @staticmethod
    def draw_blob_raw(ax, x, y, scale, colour='grey'):
        ts = np.linspace(0,2*np.pi,100)
        ax.fill(0.4*scale*np.cos(ts)+x,0.4*scale*np.sin(ts)+y,color=colour,edgecolor='none')

    def show_board(self, ax=None, matching=None, 
                   labels=False, row_labels=False, column_labels=False, 
                   symbol='rook', coloured=True,
                   marker = None, factor = None, factor_labels = None,
                   caption = None, width = None, height = None):
        """
        Displays the matching problem as a board, where the rows correspond to the left set
        and the columns correspond to the right set.  Squares corresponding to elements of
        E are shown in white, and the other squares are shown in black.

        The optional argument matching is a list of index pairs, which may or may not be
        a valid partial matching.  By default, a rook symbol is shown in each of the 
        corresponding squares.  If symbol='blob', a disc is shown instead.

        If a rook appears in a certain square, then it is not valid for a rook to appear
        in any other square in the same row or column, and all such squares are coloured
        orange.

        If a rook appears on a square that is not in E (which means that we do not have a
        valid partial matching), the square is coloured red.  Similarly, if a rook 
        appears on a square, and there is another rook in the same row or column, then
        again we do not have a valid partial matching, and the square is coloured magenta.

        If marker is not None, then it is expected to be a pair (a,b), and the corresponding
        square will be marked with a red dot.

        If factor is not None, then it is expected to be a subset of the left set, and the
        free squares in those rows will be coloured light blue, and the free squares in the
        other rows will be coloured light green.  If factor_labels is not None, then it is
        expected to be a pair of strings.  The first string will be used to label the light
        blue squares, and the second string will be used to label the light green squares.
        If factor is not None and factor_labels is None, then the strings 'C' and 'D' will
        be used as default labels.
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_axis_off()

        if matching is None:
            matching = []

        if symbol != 'blob':
            symbol = 'rook'

        if width is None:
            width = self.right_order

        if height is None:
            height = self.left_order

        T = self.classify_squares(matching)

        if factor is not None:
           for (a,b) in self.relation:
               if a in factor:
                   T[(a,b)] = "first_factor"
               else:
                   T[(a,b)] = "second_factor"

        if coloured:
            colours = {
                "free" : "white",
                "banned" : "black",
                "occupied" : "white",
                "blocked" : "orange",
                "ban_broken" : "red",
                "block_broken" : "magenta",
                "first_factor" : "#b2b2ff",
                "second_factor" : "#b2ffb2"
            }
        else:
            colours = {
                "free" : "white",
                "banned" : "black",
                "occupied" : "white",
                "blocked" : "white",
                "ban_broken" : "black",
                "block_broken" : "white",
                "first_factor" : "white",
                "second_factor" : "white"
            }

        for i in range(self.left_order):
            for j in range(self.right_order):
                a = self.left_set[i]
                b = self.right_set[j]
                type = T[(a,b)]
                colour = colours[type]
                ax.fill([j,j,j+1,j+1],[-i,-i-1,-i-1,-i],color=colour)

        for j in range(self.left_order+1):
            ax.plot([0,self.right_order],[-j,-j],color='black')
            if j < self.left_order and (labels or row_labels):
                ax.text(-0.2,-j-0.5,str(self.left_set[j]),fontsize=10,ha='right',va='center')

        for i in range(self.right_order+1):
            ax.plot([i,i],[0,-self.left_order],color='black')
            if i < self.right_order and (labels or column_labels):
                ax.text(i+0.5,0.2,str(self.right_set[i]),fontsize=10,ha='center',va='bottom')

        for (a,b) in matching:
            i = self.left_index[a]
            j = self.right_index[b]
            if symbol == 'rook':
                MatchingProblem.draw_rook_raw(ax,j+0.5,-i-0.5,1,'black')
            elif symbol == 'blob':
                MatchingProblem.draw_blob_raw(ax,j+0.5,-i-0.5,1,'black')

        if marker is not None:
            a,b = marker
            i = self.left_index[a]
            j = self.right_index[b]
            MatchingProblem.draw_blob_raw(ax,j+0.5,-i-0.5,0.3,'red')

        if factor is not None:
            if factor_labels is None:
                factor_labels = ('C','D')
            for (a,b) in self.relation:
                i = self.left_index[a]
                j = self.right_index[b]
                l = factor_labels[0] if a in factor else factor_labels[1]
                ax.text(j+0.5,-i-0.5,l,fontsize=10,ha='center',va='center')

        if width > self.right_order or height > self.left_order:
            ax.plot([width],[-height],".w")

        if caption is not None:
            ax.text(0.5*width,-height-0.5,caption,fontsize=10,ha='center',va='center')

        return ax

    def show_matrix(self, ax=None, matching=None, labels=False, row_labels=False, column_labels=False, symbol='rook'):
        """
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_axis_off()

        if matching is None:
            matching = []

        ts = np.linspace(0,2*np.pi,100)
        xs = 0.3 * np.cos(ts) + 0.5
        ys = 0.3 * np.sin(ts) - 0.5

        nA = self.right_order
        nB = self.right_order
        ax.plot([0.2,0,0,0.2],[0,0,-nA,-nA],color='black')
        ax.plot([nB-0.2,nB,nB,nB-0.2],[0,0,-nA,-nA],color='black')

        for j in range(self.left_order):
            if (labels or row_labels):
                ax.text(-0.2,-j-0.5,str(self.left_set[j]),fontsize=10,ha='right',va='center')

        for i in range(self.right_order):
            if (labels or column_labels):
                ax.text(i+0.5,0.2,str(self.right_set[i]),fontsize=10,ha='center',va='bottom')

        for i in range(nA):
            for j in range(nB):
                a = self.left_set[i]
                b = self.right_set[j]
                ax.text(j+0.5,-i-0.5,str(self.relation_matrix[i,j]),ha='center',va='center')
                if (a,b) in matching:
                    ax.plot(xs+j,ys-i,color='orange')

        return ax

    def show_graph(self, ax=None, matching=None, labels=False, row_labels=False, column_labels=False, 
                   left_pos=None, right_pos=None, scale=1):
        """
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_axis_off()

        if matching is None:
            matching = []

        if left_pos is not None:
            self.left_pos = left_pos

        if right_pos is not None:
            self.right_pos = right_pos

        P = self.left_pos
        Q = self.right_pos
        P = {a : (scale*x,scale*y) for a,(x,y) in P.items()}
        Q = {b : (scale*x,scale*y) for b,(x,y) in Q.items()}

        for (a,b) in self.relation:
            x0, y0 = P[a]
            x1, y1 = Q[b]
            if (a,b) in matching:
                col = 'orange'
                ls = '-'
            else:
                col = 'green'
                ls = ':' if len(matching) > 0 else '-'
            ax.plot([x0,x1],[y0,y1],color=col,linestyle=ls,linewidth=3,zorder=2)

        ts = np.linspace(0,2*np.pi,100)
        xs = np.cos(ts)
        ys = np.sin(ts)
        for a in self.left_set:
            x = P[a][0]
            y = P[a][1]
            ax.fill(0.3*xs+x,0.3*ys+y,facecolor='white',edgecolor='red',linewidth=3,zorder=2.1)
            ax.text(x,y,str(a),fontsize=10,ha='center',va='center',zorder=2.2)

        xs = np.sign(xs) * np.abs(xs) ** 0.1
        ys = np.sign(ys) * np.abs(ys) ** 0.1
        for b in self.right_set:
            x = Q[b][0]
            y = Q[b][1]
            ax.fill(0.3*xs+x,0.3*ys+y,facecolor='white',edgecolor='blue',linewidth=3,zorder=2.1)
            ax.text(x,y,str(b),fontsize=10,ha='center',va='center',zorder=2.2)

        return ax
    
    def print_rows(self):
        for x in self.left_set:
            R = str.join(',',sorted([str(y) for y in self.coblock[x]]))
            display(Latex('$R_{' + str(x) + '} = \{' + R + '\}$'))
    
    def print_columns(self):
        for y in self.right_set:
            C = str.join(',',sorted([str(x) for x in self.block[y]]))
            display(Latex('$C_{' + str(y) + '} = \{' + C + '\}$'))

    def print_coblocks(self):
        self.print_rows()

    def print_blocks(self):
        self.print_columns()

    def rook_report(self):
        print("There is one way to place no rooks")
        for i in range(1,len(self.rook_list_by_size)):
            n = len(self.rook_list_by_size[i])
            s = MatchingProblem.print_condensed([tuple(p) for p in self.rook_list_by_size[i]])
            if i == 1:
                print(f"There are {n} ways to place one rook\n{s}\n")
            else:
                print(f"There are {n} ways to place {i} rooks\n{s}\n")
        display(Latex("The rook polynomial is $" + self.rook_polynomial_latex + "$"))

    def show_stepwise_rook_table(self):
        m = self.left_order
        L = self.stepwise_rook_list
        nL = len(L)
        L0 = []
        for i in range(nL):
            if not(i+1 < nL and len(L[i+1]) > len(L[i]) and L[i+1][:len(L[i])] == L[i]):
                r = L[i]
                r = [(str(x[0]) + str(x[1]),x[2]) for x in r]
                r += [('',False) for i in range(m - len(r))]
                L0.append(r)
        L1 = [L0[0]]
        for i in range(1,len(L0)):
            r0 = L0[i-1]
            r = L0[i]
            r = [('',False) if r[j][0] == r0[j][0] else r[j] for j in range(m)]
            L1.append(r)
        L1 = [['<u>' + x[0] + '</u>' if x[1] else x[0] for x in r] for r in L1]
        L1 = [r + ['✗' if r[-1] == '' else '✓'] for r in L1]
        L1 = str.join("\n",['<tr>' + str.join('',['<td>' + x + '</td>' for x in r]) + '</tr>' for r in L1])
        L1 = "<table>\n" + L1 + "\n</table>\n"
        L1 = '<style>td {border: 1px solid white}</style>\n' + L1
        display(HTML(L1))

    
    @staticmethod
    def print_condensed(x):
        if isinstance(x,tuple):
            return str.join('',map(MatchingProblem.print_condensed,x))
        elif isinstance(x,list):
            return '[' + str.join(',',map(MatchingProblem.print_condensed,x)) + ']'
        elif isinstance(x,set):
            return '{' + str.join(',',map(MatchingProblem.print_condensed,x)) + '}'
        else:
            return str(x)

    @staticmethod
    def rook_tree(A,B,E,r=None):
        """
        This returns a tree T encoding the possible partial matchings.  The root of the tree
        is labelled None, and every other node is labelled by a pair (a,b) in E.
        The children of a node n encode all the possible partial matchings consisting 
        of the label on n and its ancestors, together with some additional pairs 
        that are lexicographically greater than the label on n.
        """
        T = []
        for a,b in E:
            A1 = [a1 for a1 in A if a1 > a]
            B1 = [b1 for b1 in B if b1 != b]
            E1 = {(a1,b1) for a1,b1 in E if a1 > a and b1 != b}
            T.append(MatchingProblem.rook_tree(A1,B1,E1,(a,b)))
        return Tree(r,T)
    
    @staticmethod
    def stepwise_rook_tree(A,B,E,r=None):
        """
        This is similar to the rook_tree method, but it constructs a smaller tree.  
        This tree is guaranteed to contain all full matchings, but will not 
        contain all partial matchings except in trivial cases.  The nodes have
        labels (a,b,t) where (a,b) is a pair in E, and t is a boolean value, 
        which records whether the node is the last child of its parent.
        If we have a node labelled (a0,b0,t0) and a child labelled 
        (a1,b1,t1), then a1 is always the successor of a0 in A.  In the original
        rook tree method, a1 can be any element of A greater than a0.  Thus, the
        stepwise rook tree method only gives partial matchings of size k that have 
        a rook in each of the first k rows, and no rooks in the remaining rows.
        """
        T = []
        A = sorted(list(A))
        B = sorted(list(B))
        if len(A) == 0:
            return Tree(r,[])
        a = A[0]
        B0 = [b0 for (a0,b0) in E if a0 == a]
        m0 = len(B0)
        for i,b in enumerate(B0):
            A1 = [a1 for a1 in A if a1 > a]
            B1 = [b1 for b1 in B if b1 != b]
            E1 = [(a1,b1) for a1,b1 in E if a1 > a and b1 != b]
            T.append(MatchingProblem.stepwise_rook_tree(A1,B1,E1,(a,b,i==m0-1)))
        return Tree(r,T)
    
    @staticmethod
    def from_blocks(C, fast=False):
        """
        Constructs a matching problem from a dictionary C, where the keys are sets or lists
        and the values are sets or lists.  The matching problem has right set B consisting of
        the keys of C, left set A consisting of the union of the values of C, and relation E
        consisting of the pairs (a,b) such that b is in B and a is in C[b].
        """
        if not(isinstance(C,dict)):
            raise ValueError("Input must be a dictionary")
        B = C.keys()
        if not all(isinstance(C[b],set) or isinstance(C[b],list) for b in B):
            raise ValueError("Values of dictionary must be sets or lists")
        A = set()
        E = set()
        for b in B:
            A = A.union(set(C[b]))
            E = E.union({(a,b) for a in C[b]})
        return MatchingProblem(A,B,E,fast)

    @staticmethod
    def from_coblocks(R, fast=False):
        """
        Constructs a matching problem from a dictionary R, where the keys are sets or lists
        and the values are sets or lists.  The matching problem has left set A consisting of
        the keys of R, right set B consisting of the union of the values of R, and relation E
        consisting of the pairs (a,b) such that a is in A and b is in R[a].
        """
        if not(isinstance(R,dict)):
            raise ValueError("Input must be a dictionary")
        A = R.keys()
        if not all(isinstance(R[a],set) or isinstance(R[a],list) for a in A):
            raise ValueError("Values of dictionary must be sets or lists")
        B = set()
        E = set()
        for a in A:
            B = B.union(set(R[a]))
            E = E.union({(a,b) for b in R[a]})
        return MatchingProblem(A,B,E,fast)
    
    @staticmethod
    def from_matrix(M, fast=False, marker=1):
        """
        Constructs a matching problem from a matrix M, where M is a numpy array.
        The matching problem has left set A consisting of the rows of M, right set B
        consisting of the columns of M, and relation E consisting of the pairs (a,b) such
        that M[a,b] = marker.
        """
        if not(isinstance(M,np.ndarray) and M.ndim == 2):
            raise ValueError("Input must be a numpy array")
        A = list(range(M.shape[0]))
        B = list(range(M.shape[1]))
        E = {(a,b) for a in A for b in B if M[a,b] == marker}
        return MatchingProblem(A,B,E,fast)
    
    @staticmethod
    def full_board(A, B=None):
        if isinstance(A,int):
            A = list(range(A))
        if isinstance(B,int):
            B = list(range(B))
        if B is None:
            B = A.copy()
        AB = [(a,b) for a in A for b in B]
        return MatchingProblem(A, B, AB)
    
    @staticmethod
    def linear_board(n):
        return MatchingProblem.full_board(1,n)
    
    @staticmethod
    def diagonal_board(n):
        return MatchingProblem(range(n),range(n),[(j,j) for j in range(n)])
    
    @staticmethod
    def staircase_board(n):
        return MatchingProblem(range(n),range(n),[(0,0)] + [(i,j) for i in range(1,n) for j in range(i-1,i+1)])

    @staticmethod
    def augmented_staircase_board(n):
        return MatchingProblem(range(n),range(n),[(0,0),(0,n-1)] + [(i,j) for i in range(1,n) for j in range(i-1,i+1)])
    
    @staticmethod
    def quadratic_residue_design(p):
        if not (isinstance(p,int) and sp.isprime(p) and p % 4 == 3):
            raise ValueError("p must be a prime number congruent to 3 mod 4")
        n = (p-3) // 4
        A = list(range(-(2*n+1),2*n+2))
        B = A.copy()
        def mod(x):
            y = x % p
            return y if y <= p // 2 else y - p
        Q = {mod(i * i) for i in range(1,2*n+2)}
        Q = sorted(list(Q))
        E = {(i,j) for i in A for j in B if mod(i-j) in Q}
        return MatchingProblem(A,B,E,fast=True)


