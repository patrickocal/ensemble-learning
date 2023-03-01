from itertools import combinations, permutations
from math import comb, factorial
import numpy as np
import pandas as pd
import pickle 
# number of maturity dates
m = 8
sub = 3
subch2 = comb(sub, 2)
s = "data/" + str(m) + "choose" + str(sub)
# generate indices for subsets of four maturity dates
a = np.array(list(combinations(range(1, m), sub)))

rankings = pd.read_csv(s + "rankings.csv")
countof = pd.read_csv(s + "countof.csv")
num_days = len(rankings)
holds_prop = {}
p3d_test = {}
f3d_test = {}
urankings = {}
for i in a:
    urankings[str(i)]\
    = [np.zeros(subch2, dtype=int) for x in range(factorial(sub))]
print(urankings[str(a[0])])
def get_diversity_pts(subsets, rankdf, fin, start=0):
    fin = len(rankdf)
    dd = {}
    for i in subsets:
        dd[str(i)] = {}
        dd[str(i)][start] = rankdf[str(i)][start]
        for j in range(start + 1, fin):
            if rankdf[str(i)][j] != rankdf[str(i)][j - 1]:
                dd[str(i)][j] = rankdf[str(i)][j]
#    d = [(i, j) for i in subsets for j in range(start, fin)
#         if rankdf[str(i)][j] != rankdf[str(i)][j-1]]
#    dd = {}
#    [dd[str(i)].append(j) for (i, j) in d]
    with open(s + "diversity_pts.pkl", "wb") as f:
        pickle.dump(dd, f)
    return dd
#get_diversity_pts(a, rankings, len(rankings)) # uncomment to generate div_pts
with open (s + "diversity_pts.pkl", "rb") as f:
    div_pts = pickle.load(f)
print(len(div_pts[str(a[0])]))
print(div_pts[str(a[0])])
acountof = countof
b = {}
listtobin = {}
def ranktopos(rankvec, natord):
    pos = np.zeros(len(rankvec), dtype=int)
    for k in range(len(rankvec)):
        for l in range(len(natord)):
            if rankvec[k] == natord[l]:
                pos[l] = k
    return pos 
pos = ranktopos([7, 5, 3, 1], [1, 3, 5, 7])
print(pos)
#print(list(pos.keys())[1]) 
def ranktoposinv(posvec, natord):
    rank = np.zeros(len(natord), dtype=int)
    for k in range(len(posvec)):
            rank[posvec[k]] = natord[k]
    return rank
rank = ranktoposinv([3, 2, 1, 0], [1, 3, 5, 7])
print(rank)

def bintopos(binvec, natord):
    pos = np.zeros(len(natord), dtype=int)
    while sum(pos) < subch2:
        for k in range(len(binvec)):
            for l in range(len(natord)):
                if binvec[k] == natord[l]:
                    pos[l] += 1
    return pos
pos = bintopos([1, 1, 1, 3, 3, 5], [1, 3, 5, 7])
print(pos)
def bintoposinv(posvec, natord):
    comb = list(combinations(range(len(posvec)), 2))
    binvec = np.zeros(len(comb), dtype=int)
    l = 0
    for k in comb:
        if posvec[k[0]] < posvec[k[1]]:
            binvec[l] = natord[k[1]]
        else:
            binvec[l] = natord[k[0]]
        l += 1
    return binvec
binvec = bintoposinv([3, 2, 1, 0], [1, 3, 5, 7])
print(binvec)

def ranktobin(rankvec, natord):
    posvec = ranktopos(rankvec, natord)
    binvec = bintoposinv(posvec, natord)
    return binvec

def bintorank(binvec, natord):
    posvec = bintopos(binvec, natord)
    rankvec = ranktoposinv(posvec, natord)
    return rankvec

def ktdistancebin(p, q):
    p = np.array(p)
    q = np.array(q) if len(p) == len(q) and len(p) > 0 else print("ERROR: len")
    d = sum(p != q)
    return d
dist = ktdistancebin([1, 1, 2], [2, 3, 3])
print(dist)

def strtovec(s):
    s = s.replace("[", "")
    s = s.replace("]", "")
    v = [int(n) for n in s.split()]
    return np.array(v)
v = strtovec(str(np.zeros(3, dtype=int)))
print(v)
#for i in a:
#    listtobin[str(i)] = {}
#    for h in perm:
#        listtobin[str(i)][h] = {}#[None]*subch2
#        inc = 0
#        for k in list(combinations(i, 2)):
#            [x, y] = [k[0], k[1]]
#            listtobin[str(i)][h][k] = max(h(x), h(y))
#            inc += 1

b = {}
for i in a:
    ldi = len(div_pts[str(i)])
    perm = list(permutations(i))
    b[str(i)] = {}
    for h in perm:
        b[str(i)][str(h)] = 0 # h is initially absent from the list of rankings
    for j in range(ldi): # start the main loop for i
        lu = len(urankings[str(i)])
        ak = acountof[str(i)][j]
print(lu)
print(urankings[str(a[0])])
#        if (lu = 1):
#            urankings[str(i)][1]\
#            = [int(n) for n in rankings[str(i)][j].split()]
#            urankings[str(i)][1][k] != urankings[str(i)][0][k]:
#                    b[1][k] += 1
#            ak = lu + b[1][k] 
#        elif (lu = 2 and ak = 2):
#            urankings[str(i)][2]\
#            = [int(n) for n in rankings[str(i)][j].split()]
#            for k in range(subch2):
#                if urankings[str(i)][2][k] != urankings[str(i)][0][k]:
#                    b[2][k] += 1
#                if urankings[str(i)][2][k] != urankings[str(i)][1][k]:
#                    b[2][k] += 1
#            if max(b) = 3:
#                ak = 4
#        elif (lu = 3):
#            urankings[str(i)][3]\
#            = [int(n) for n in rankings[str(i)][j].split()]
#            b = [0, 0, 0]
#            for k in range(subch2):
#                if urankings[str(i)][3][k] != urankings[str(i)][0][k]:
#                    b[0] += 1
#                if urankings[str(i)][3][k] != urankings[str(i)][1][k]:
#                    b[1] += 1
#                if urankings[str(i)][3][k] != urankings[str(i)][2][k]:
#                    b[1] += 1
#            ak = lu + min(b)
#    
#        
#        
#            if (countof[str(i)][j] = 2) and (rankings[str(i)][j] !=
#                                                rankings[str(i)][j - 1]):
#                urankings[str(i)][2] = rankings[str(i)][j]
#        
#                p3d_test[str(i)] = j + 1
#            if countof[str(i)][j] < 5:
#                f3d_test[str(i)] = j + 1
#        holds_prop['p3d'] = [1 - p3d_test[max(p3d_test)] / num_days]
#        holds_prop['f3d'] = [1 - f3d_test[max(f3d_test)] / num_days]
#        holds_propd = pd.DataFrame(holds_prop)
#        holds_propd.to_csv(s + "holds_prop.csv")
