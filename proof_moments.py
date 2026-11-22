"""Counts for the rooted triangle terms in the offer inequalities."""

from itertools import combinations, permutations
from functools import lru_cache


PAIRS=tuple(combinations(range(6),2))
ORDERINGS=tuple(permutations(range(6)))


@lru_cache(maxsize=None)
def counts(mask):
    """Count ordered configurations in the complement of the input graph."""
    adj=[[0]*6 for _ in range(6)]
    for i,(x,y) in enumerate(PAIRS):
        adj[x][y]=adj[y][x]=1-((mask>>i)&1)
    tri=bow=star=star_tri=k4=k4_tri=0
    for x,y,z,p,q,r in ORDERINGS:
        if not (adj[x][y] and adj[x][z]):
            continue
        if adj[y][z]:
            tri+=1
            bow+=adj[x][p]*adj[x][q]*adj[p][q]
        if adj[x][p]:
            star+=1
            extra=adj[x][q]*adj[x][r]*adj[q][r]
            star_tri+=extra
            if adj[y][z] and adj[y][p] and adj[z][p]:
                k4+=1
                k4_tri+=extra
    return tri,bow,star,star_tri,k4,k4_tri
