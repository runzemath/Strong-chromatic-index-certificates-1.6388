"""Integer coefficients of additional local flag inequalities."""

from collections import defaultdict
from itertools import combinations, permutations


VERTICES = tuple(range(5))
ORDERS = tuple(permutations(VERTICES,2))
TRIPLES = tuple(combinations(VERTICES,3))
QUADRUPLES = tuple(combinations(VERTICES,4))


def local(vertices,colors,adj):
    reached={v for v in vertices if colors[v]==0}
    for _ in vertices:
        reached.update(v for v in vertices if any(adj[v][w] for w in reached.copy()))
    return len(reached)==len(vertices)


def key(vertices,colors,matrix):
    return min(tuple(colors[v] for v in order)+tuple(matrix[a][b] for a,b in combinations(order,2))
               for order in permutations(vertices))


def coefficients(colors,matrix):
    """Each key gives the coefficient p - tau*q on the current flag."""
    result=defaultdict(lambda:[0,0])
    adj=[[int(v>0) for v in row] for row in matrix]

    def add(k,p,q=0):
        if p or q:
            result[k][0]+=p
            result[k][1]+=q

    for x,y in ORDERS:
        if colors[y]!=0 or matrix[x][y]!=1:
            continue
        rem=[v for v in VERTICES if v not in (x,y)]
        for w in rem:
            if colors[w]==1 and not (adj[x][w] or adj[y][w]):
                continue
            u,v=[a for a in rem if a!=w]
            in_u=adj[x][u] or adj[y][u]
            in_v=adj[x][v] or adj[y][v]
            both_x=adj[x][u]*adj[x][v]
            pattern=(colors[x],colors[w],matrix[x][w],matrix[y][w])
            add(('degree',*pattern),2*int(matrix[u][v]==1 and (in_u or in_v)),8*both_x)
            codeg=both_x*(adj[y][u]+adj[y][v])
            internal=adj[u][v]*in_u*in_v
            unmarked=int(matrix[u][v]==2 and (in_u or in_v))
            add(('deficit',*pattern),4*both_x-codeg-internal-unmarked,4*both_x)

    for vertices in QUADRUPLES:
        v=next(a for a in VERTICES if a not in vertices)
        if local(vertices,colors,adj):
            p=4*int(colors[v]==0)
            q=2*sum(adj[v][w] for w in vertices)
            if p or q:
                add(('black',*key(vertices,colors,matrix)),p,q)

    for vertices in TRIPLES:
        u,v=[a for a in VERTICES if a not in vertices]
        if not local(vertices,colors,adj):
            continue
        incidences=(2-colors[u]-colors[v])*int(matrix[u][v]==1)
        count=(1-colors[u])*sum(adj[v][w] for w in vertices)
        count+=(1-colors[v])*sum(adj[u][w] for w in vertices)
        if count!=3*incidences:
            add(('incidence',*key(vertices,colors,matrix)),count-3*incidences)
    return result
