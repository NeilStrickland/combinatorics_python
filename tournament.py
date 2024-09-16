import numpy as np
import matplotlib.pyplot as plt
import math
from IPython.display import display, Latex
from matching_problem import MatchingProblem

class ScoreSystem():
    def __init__(self,scores,players=None):
        if isinstance(scores,dict):
            self.scores_by_player = scores
            self.players = list(scores.keys())
            self.scores_list = [scores[p] for p in self.players]
            self.sorted_scores = sorted(self.scores_list,reverse=True)
        elif isinstance(scores,list):
            self.scores_list = scores
            if players is None:
                self.players = list(range(len(scores)))
            else:
                self.players = players
            self.scores_by_player = {p : s for p,s in zip(self.players,scores)}
            self.sorted_scores = sorted(scores,reverse=True)
        n = len(self.players)
        s = self.sorted_scores
        self.landau_lower_tests = []
        for k in range(1,n+1):
            st = s[-k:]
            ss = sum(st)
            bn = math.comb(k,2)
            ok = ss >= bn if k < n else ss == bn
            self.landau_lower_tests.append((st,ss,bn,ok))
        self.landau_upper_tests = []
        for k in range(1,n+1):
            st = s[:k]
            ss = sum(st)
            bn = math.comb(k,2) + k * (n - k)
            ok = ss <= bn if k < n else ss == bn
            self.landau_upper_tests.append((st,ss,bn,ok))
        self.is_plausible = all([ok for (st,ss,bn,ok) in self.landau_lower_tests])
        At = {}
        Bt = {}
        C = {}
        for i in range(n):
            for j in range(i+1,n):
                At[(i,j)] = str(self.players[i]) + 'v' + str(self.players[j])
        for i in range(n):
            for k in range(self.scores_list[i]):
                Bt[(i,k)] = str(self.players[i]) + '|' + str(k)
                C[Bt[(i,k)]] = [At[(x,i)] for x in range(i)] + [At[(i,x)] for x in range(i+1,n)]
        A = list(At.values())
        B = list(Bt.values())
        E = [(a,b) for b in Bt.values() for a in C[b]]
        self.matching_problem = MatchingProblem(A,B,E,fast=True)

    def landau_lower_report(self):
        n = len(self.players)
        all_ok = True
        for k in range(1,n+1):
            st, ss, bn, ok = self.landau_lower_tests[k-1]
            all_ok = all_ok and ok
            s = 'The test $' + '+'.join([str(x) for x in st]) + '=' + str(ss) + \
                ('\geq' if k < n else '=') + '\\binom{' + str(k) + '}{2}=' + str(bn) + '$ is ' + \
                ('satisfied' if ok else 'not satisfied')
            display(Latex(s))
        if all_ok:
            display(Latex('All tests are satisfied, so Landau\'s theorem guarantees the existence of a tournament with these scores'))
        else:
            display(Latex('Not all tests are satisfied, so there cannot be a tournament with these scores'))

class Tournament():
    def __init__(self, R):
        R = sorted(list(R))
        if not(isinstance(R, list)): 
            raise ValueError("R must be convertible to a list")
        if not(all([isinstance(x, tuple) and len(x) == 2 for x in R])): 
            raise ValueError("R must be a list of pairs")
        P = sorted(list(set([x for y in R for x in y])))
        self.players = P
        nP = len(P)
        self.num_players = nP
        self.player_index = {p : i for i, p in enumerate(P)}
        X1 = sorted(R + [(y, x) for (x, y) in R])
        X2 = sorted([(x, y) for y in P for x in P if x != y])
        if X1 != X2:
            raise ValueError("R is not a valid result list")
        self.results = R
        self.win_list = {}
        self.score = {}
        for p in P:
            self.win_list[p] = [y for (x,y) in R if x == p]
            self.score[p] = len(self.win_list[p])
        self.score_system = ScoreSystem(self.score)
        self.sorted_scores = sorted([self.score[p] for p in P], reverse=True)
        self.outcome = {}
        self.indexed_outcome = {}
        self.indexed_results = []
        for i in range(nP):
            for j in range(nP):
                p = P[i]
                q = P[j]
                if i == j:
                    self.outcome[(p, q)] = 0
                elif (p, q) in R:
                    self.outcome[(p, q)] = -1
                    self.indexed_results.append((i, j))
                else:
                    self.outcome[(p, q)] = 1
                self.indexed_outcome[(i, j)] = self.outcome[(p, q)]
        self.pos = None

    def find_winning_line(self):
        p = [self.players[0]]
        for i in range(1,self.num_players):
            x = self.players[i]
            j = 0
            while j < len(p) and self.outcome[(x,p[j])] == 1:
                j += 1
            p.insert(j,x)
        return p

    def show_table(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.axis('off')
        n = self.num_players
        ax.plot([0,n],[1,1],'k-')
        ax.plot([-1,n],[0,0],'k-',linewidth=3)
        ax.plot([-1,-1],[0,-n],'k-')
        ax.plot([0,0],[1,-n],'k-',linewidth=3)
        for i in range(n):
            ax.plot([-1,n],[-i-1,-i-1],'k-')
            ax.plot([i+1,i+1],[1,-n],'k-')
            ax.fill([i,i+1,i+1,i,i],[-i,-i,-i-1,-i-1,-i],'k')
            ax.text(i+0.5,0.5,self.players[i],ha='center',va='center')
            ax.text(-0.5,-i-0.5,self.players[i],ha='center',va='center')

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                u = self.indexed_outcome[(i,j)]
                if u == 1:
                    c = '#b2b2ff'
                    s = 'W'
                else:
                    c = '#b2ffb2'
                    s = 'L'
                ax.fill([i,i+1,i+1,i,i],[-j,-j,-j-1,-j-1,-j],c)
                ax.text(i+0.5,-j-0.5,s,ha='center',va='center')

        return ax

    def show_graph(self, ax=None, pos=None, line = None):
        n = self.num_players
        if n == 0:
            ValueError("No players in the tournament")
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.axis('off')
        if pos is not None:
            self.pos = pos
        if self.pos is None:
            self.pos = {} 
            for i in range(n):
                p = self.players[i]
                t = 2 * np.pi * i / n + np.pi / 2
                self.pos[p] = (np.cos(t), np.sin(t))
        for i in range(n):
            p = self.players[i]
            ax.text(self.pos[p][0], self.pos[p][1], p, ha='center', va='center')
        z = 1
        if line is not None:
            line_steps = [(line[i],line[i+1]) for i in range(len(line)-1)]
        else:
            line_steps = None
        for i in range(n):
            for j in range(i+1, n):
                p = self.players[i]
                q = self.players[j]
                a = self.pos[p]
                b = self.pos[q]
                if self.indexed_outcome[(i,j)] == 1:
                    a, b = b, a
                    p, q = q, p
                e = 0.05
                a0 = (a[0] + e * (b[0] - a[0]), a[1] + e * (b[1] - a[1]))
                b0 = (b[0] - e * (b[0] - a[0]), b[1] - e * (b[1] - a[1]))
                u0 = (b0[0] - a0[0], b0[1] - a0[1])
                ax.plot([a0[0], b0[0]], [a0[1], b0[1]], 'w-', linewidth=9, zorder=z)
                z += 0.05
                if line_steps is None:
                    c = 'black'
                    lw = 1
                elif (p,q) in line_steps:
                    c = 'blue'
                    lw = 2
                else:
                    c = 'grey'
                    lw = 1
                ax.arrow(*a0, *u0, color=c, linewidth = lw, 
                         head_length=0.03, head_width=0.03, length_includes_head=True, zorder=z)
                z += 0.05
        return ax
    
    def show_medals(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.axis('off')
        n = self.num_players
        if n == 0:
            ValueError("No players in the tournament")
        s = self.sorted_scores[0]
        ax.plot([0,2*n],[1,1],'k-')
        ax.plot([0,2*n],[0,0],'k-')
        ax.plot([0,2*n],[-2*s,-2*s],'k-')
        for i in range(n+1):
            ax.plot([2*i,2*i],[1,-2*s],'k-')
        ts = np.linspace(0,2*np.pi,100)
        xs = 0.55 * np.cos(ts)
        ys = 0.55 * np.sin(ts)
        for i in range(n):
            p = self.players[i]
            ax.text(2*i+1,0.5,p,ha='center',va='center')
            for j, q in enumerate(self.win_list[p]):
                ax.fill(2*i+1 + xs, -2*j-1 + ys, '#e8b923')
                t = p + ' v ' + q if p < q else q + ' v ' + p
                ax.text(2*i+1,-2*j-1,t,ha='center',va='center')
        return ax
    
    @staticmethod
    def consistent_tournament(P):
        if isinstance(P, int):
            P = list(range(P))
        else:
            P = list(P)
        T = [(P[i],P[j]) for i in range(len(P)) for j in range(i+1,len(P))] 
        return Tournament(T)
    
    @staticmethod
    def odd_modular_tournament(n):
        if not(isinstance(n,int) and n > 0 and n % 2 == 1):
            raise ValueError("n must be a positive odd integer")
        P = list(range(n))
        m = (n-1) // 2
        T = [(i,(i+j) % n) for j in range(1,m+1) for i in range(n)] 
        return Tournament(T)
    
    @staticmethod
    def colon(T,U):
        if not(isinstance(T,Tournament) and isinstance(U,Tournament)):
            raise ValueError("Arguments must be tournaments")
        X = set(T.players) & set(U.players)
        if len(X) == 0:
           P = T.players + U.players
           R = T.results + U.results + [(x,y) for x in T.players for y in U.players]
           return Tournament(R)
        else:
            n = T.num_players
            m = U.num_players
            R = T.indexed_results + [(i,j) for i in range(n) for j in range(n,m+n)] + \
                [(i+n,j+n) for (i,j) in U.indexed_results]
            return Tournament(R)
            
    @staticmethod
    def star(T,U):
        if not(isinstance(T,Tournament) and isinstance(U,Tournament)):
            raise ValueError("Arguments must be tournaments")
        n = T.num_players
        m = U.num_players
        if T.players == list(range(n)) and U.players == list(range(n,n+m)):
            R = []
            for i0 in range(n):
                for j0 in range(m):
                    for i1 in range(n):
                        for j1 in range(m):
                            k0 = i0 + j0 * n
                            k1 = i1 + j1 * n
                            if (i0,i1) in T.indexed_results or (i0 == i1 and (j0,j1) in U.indexed_results):
                                R.append((k0,k1))
            return Tournament(R)
        else:
            R = []
            for t0 in T.players:
                for u0 in U.players:
                    for t1 in T.players:
                        for u1 in U.players:
                            x0 = str(t0) + str(u0)
                            x1 = str(t1) + str(u1)
                            if (t0,t1) in T.results or (t0==t1 and (u0,u1) in U.results):
                                R.append((x0,x1))
            return Tournament(R)

        