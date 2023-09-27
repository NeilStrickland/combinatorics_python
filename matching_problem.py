import numpy as np

class matching_problem:
    def __init__(self,A,B,E,fast=False):
        if isinstance(A,int):
            nA = A
            A = list(range(A))
        elif isinstance(A,set):
            A = list(A)
            nA = len(A)
        elif isinstance(A,list):
            A = A
            nA = len(A)
        else:
            raise ValueError("A must be an int, set, or list")
        
        if isinstance(B,int):
            nB = B
            B = list(range(B))
        elif isinstance(B,set):
            B = list(B)
            nB = len(B)
        elif isinstance(B,list):
            B = B
            nB = len(B)
        else:
            raise ValueError("B must be an int, set, or list")
        
        if isinstance(E,list) or isinstance(E,set):
            E = set(E)
        else:
            raise ValueError("E must be a list or set")
        
        AB = {(a,b) for a in A for b in B}

        if E.difference(AB) != set():
            raise ValueError("E must be a subset of A x B")
        
        self.left_set = A
        self.right_set = B
        self.left_order = nA
        self.right_order = nB
        self.product_set = AB
        self.relation = E
        self.complement = AB.difference(E)
        self.block = {}
        self.coblock = {}
        self.relation_table = {}
        self.index = {}
        self.left_index = {}
        self.right_index = {}

        for i in range(nA):
            self.left_index[A[i]] = i
        for i in range(nB):
            self.right_index[B[i]] = i

        for i in range(nA):
            for j in range(nB):
                self.index[(A[i],B[j])] = (i,j)
        
        for a in A:
            self.block[a] = {b for b in B if (a,b) in E}

        for b in B:
            self.coblock[b] = {a for a in A if (a,b) in E}

        self.indexed_relation = {(self.left_index[a],self.right_index[b]) for a,b in E}
        self.indexed_complement = {(self.left_index[a],self.right_index[b]) for a,b in AB.difference(E)}

        M = np.zeros((nA,nB))
        for a,b in E:
            M[self.left_index[a],self.right_index[b]] = 1
        self.relation_table = M

        if not(fast):
            self.full_init()

    def full_init(self):
        A1 = list(range(self.left_order))
        B1 = list(range(self.right_order))
        T1 = self.rook_tree(A1,B1,self.indexed_relation)
        L1 = self.rook_list_from_tree(T1)
        p1 = np.zeros(min(self.left_order,self.right_order))
        for m in L1:
            p1[len(m)] += 1
        self.indexed_rook_tree = T1
        self.indexed_rook_list = L1
        self.rook_plynomial = p1
        self.indexed_matchings = [m for m in L1 if len(m) == self.left_order]
        self.indexed_comatchings = [m for m in L1 if len(m) == self.right_order]
        f = lambda v : (self.left_set[v[0]],self.right_set[v[1]])
        self.rook_list   = [list(map(f,u)) for u in self.indexed_rook_list]
        self.matchings   = [list(map(f,u)) for u in self.indexed_matchings]
        self.comatchings = [list(map(f,u)) for u in self.indexed_comatchings]
        self.num_matchings = len(self.matchings)
        self.num_comatchings = len(self.comatchings)
        m = max(len(u) for u in self.rook_list)
        self.maximal_matching_size = m
        self.maximal_matchings = [u for u in self.rook_list if len(u) == m]
        self.plausible_sets = []
        self.implausible_sets = []
        self.tricky_sets = []
        self.cuts = []
        P = [set()]
        for a in list(self.left_set):
            P.extend([p.union({a}) for p in P])
        self.left_power_set = P
        for S in P:
            T = set()
            for a in S:
                T = T.union(self.block[a])
            if len(T) < len(S):
                self.implausible_sets.append(S)
            else:
                self.plausible_sets.append(S)
                if len(T) == len(S):
                    self.tricky_sets.append(S)
            Sc = self.left_set.difference(S)
            self.cuts.append((Sc,T))
        self.plausible_cosets = []
        self.implausible_cosets = []
        self.tricky_cosets = []
        Q = [set()]
        for b in list(self.right_set):
            Q.extend([q.union({b}) for q in Q])
        self.right_power_set = Q
        for T in Q:
            S = set()
            for b in T:
                S = S.union(self.coblock[b])
            if len(S) < len(T):
                self.implausible_cosets.append(S)
            else:
                self.plausible_cosets.append(S)
                if len(S) == len(T):
                    self.tricky_cosets.append(S)
        
    def check_design(self):
        self.design_v = self.right_order
        self.design_b = self.left_order
        kk = {len(self.block[a]) for a in self.left_set}
        if len(kk) == 1:
            self.design_k = kk.pop()
        else:
            self.design_k = None
        rr = {len(self.coblock[b]) for b in self.right_set}
        if len(rr) == 1:
            self.design_r = rr.pop()
        else:
            self.design_r = None
        ll = []
        for i in range(self.right_order):
            for j in range(i):
                ll.append(len(self.coblock[i].intersection(self.coblock[j])))
        if len(set(ll)) == 1:
            self.design_lambda = ll.pop()
        else:
            self.design_lambda = None
        if self.design_k is None or self.design_r is None or self.design_lambda is None:
            self.is_design = False
            self.design_params = None
        else:
            self.is_design = True
            self.design_params = (self.design_v,self.design_b,self.design_r,self.design_k,self.design_lambda)
        return self.is_design

    def rook_tree(self,A,B,E):
        if len(E) == 0:
            return []
        T = []
        for a,b in E:
            A1 = [a1 for a1 in A if a1 > a]
            B1 = [b1 for b1 in B if b1 != b]
            E1 = {(a1,b1) for a1,b1 in E if a1 > a and b1 != b}
            T.append([(a,b), *self.rook_tree(A1,B1,E1)])
        return T
    
    def rook_list_from_tree(self,T):
        L = []
        for x in T:
            a, *t = x
            if len(t) == 0:
                L.append([a])
            else:
                M = self.rook_list_from_tree(t)
                L.extand([(a,*m) for m in M])
        return L

