
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd


# number of maturity dates
m = 9
sub = 3
subch2 = comb(sub, 2)
# generate indices for subsets of four maturity dates
a = np.array(list(combinations(range(1,m), sub)))

print(a[1])
print(len(a))

sveny = pd.read_csv("data/FED-SVENY-20230224.csv")
num_days = 10#min(sveny.count())#int(sveny.shape[0]/1)
print(sveny.head(10))
print(num_days)

print(sveny.iloc[1:10, a[0]])
d = {}
for i in a:
    d[str(i)] = ["[" for x in range(num_days)]
for i in a:
    c = np.array(list(combinations(i, 2)))
    for j in range(num_days):
        for k in range(subch2):
            b = np.array(sveny.iloc[j, c[k]])
         #   print(b)
            if b[0] < b[1]:
                d[str(i)][j] += " " + str(c[k][1])
            else:
                d[str(i)][j] += " " + str(c[k][0])
        d[str(i)][j] += "]"
        d[str(i)][j] = d[str(i)][j].replace(" ", "", 1)
r = pd.DataFrame(d)
r.rename(sveny['Date'], axis='index', inplace =True)
s = "data/" + str(m) + "choose" + str(sub) 
r.to_csv(s + "rankings.csv")

countof = {}
for i in a:
    countof[str(i)] = np.zeros(num_days, dtype=int)
    for j in range(num_days):
        countof[str(i)][j] = r[str(i)][0:j].nunique()

countof = pd.DataFrame(countof)
countof.to_csv(s + "countof.csv")
