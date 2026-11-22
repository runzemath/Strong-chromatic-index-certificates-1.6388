"""Ordinary graph flags for the pair, triple and four-set occupancy estimate."""

from fractions import Fraction as Q
from functools import lru_cache
from itertools import combinations, permutations
from math import factorial, lcm
from pathlib import Path
import json

from verify import check_positive_definite

here = Path(__file__).resolve().parent
order = 6
triple_polynomial = {
    (0,0): Q(1), (1,0): Q(1), (0,1): Q(1), (2,0): Q(1),
    (1,1): Q(1), (0,2): Q(1), (3,0): Q(6), (2,1): Q(3),
    (1,2): Q(3), (0,3): Q(3),
}
four_linear = Q(23,12)
four_parameters = [Q(1,2),Q(2,3),Q(3,4)]
four_quadratic = sum(a*a for a in four_parameters) + sum(a*b for a,b in combinations(four_parameters,2))
pair_coeff = [Q(1,2**(i+1)) for i in range(5)]
triple_coeff = {ab: x / (6 * 2**ab[0] * 3**ab[1]) for ab,x in triple_polynomial.items()}
four_coeff = [Q(1,30),four_linear/180,four_quadratic/1080]
denominator = lcm(*(x.denominator for x in [*pair_coeff,*triple_coeff.values(),*four_coeff]))
pc = [int(x*denominator) for x in pair_coeff]
tc = {ab:int(x*denominator) for ab,x in triple_coeff.items()}
fc = [int(x*denominator) for x in four_coeff]

@lru_cache(maxsize=None)
def pairs(n):
    return tuple(combinations(range(n),2))

@lru_cache(maxsize=None)
def unlabel(n, roots):
    """Return canonical representatives and their indices on all bitmasks."""
    ij=pairs(n)
    ids={pair:i for i,pair in enumerate(ij)}
    changes=[]
    for p in permutations(range(roots,n)):
        p=tuple(range(roots))+p
        changes.append(tuple(1<<ids[tuple(sorted((p[i],p[j])))] for i,j in ij))
    representatives=[]
    table=[-1]*(1<<len(ij))
    for mask in range(len(table)):
        if table[mask]>=0:
            continue
        ix=len(representatives)
        representatives.append(mask)
        present=[k for k in range(len(ij)) if (mask>>k)&1]
        for perm in changes:
            image=sum(perm[k] for k in present)
            table[image]=ix
    assert min(table)>=0
    return representatives, table

def subgraph(mat, vertices):
    return sum(int(mat[vertices[i]][vertices[j]])<<k for k,(i,j) in enumerate(pairs(len(vertices))))

def polynomial(mat, vertices):
    def edge(i,j):
        return mat[vertices[i]][vertices[j]]
    if not edge(0,1):
        return 0
    b=[(1-edge(0,k))*(1-edge(1,k)) for k in range(2,6)]
    ans=pc[0]
    prod=1
    for i in range(4):
        prod*=b[i]
        ans+=pc[i+1]*prod
    if not(edge(0,2) and edge(1,2)):
        return ans
    x=b[1:]
    y=[max(0,2-edge(0,k)-edge(1,k)-edge(2,k)) for k in range(3,6)]
    for (a,beta),val in tc.items():
        prod=1
        for i in range(a):
            prod*=x[i]
        for i in range(a,a+beta):
            prod*=y[i]
        ans-=val*prod
    if not(edge(0,3) and edge(1,3) and edge(2,3)):
        return ans
    u=4-sum(edge(i,4) for i in range(4))
    v=4-sum(edge(i,5) for i in range(4))
    u=u*(u-1)//2
    v=v*(v-1)//2
    return ans+fc[0]+fc[1]*u+fc[2]*u*v

@lru_cache(maxsize=1)
def flag_data():
    pos=list(pairs(order))
    reps, _ = unlabel(order,0)
    block_specs=[]
    by_type={}
    for roots in (0,2,4):
        ext_size=(order+roots)//2
        ext_reps, ext_tab=unlabel(ext_size,roots)
        type_reps, _ = unlabel(roots,0)
        for t in type_reps:
            ext_ids={}
            for i,ext in enumerate(ext_reps):
                type_of_ext=sum(((ext>>pairs(ext_size).index((u,v)))&1)<<j
                                for j,(u,v) in enumerate(pairs(roots)))
                if type_of_ext==t:
                    ext_ids[i]=len(ext_ids)
            block=len(block_specs)
            by_type[roots,t]=(block,ext_ids)
            block_specs.append((roots,t,len(ext_ids)))
    permutations_of_order=list(permutations(range(order)))
    fourth_reps,fourth_tab=unlabel(4,0)
    types_data={roots:unlabel((order+roots)//2,roots)[1] for roots in (0,2,4)}
    matrices=[[[[0]*d for _ in range(d)] for _ in reps] for _,_,d in block_specs]
    objective=[0]*len(reps)
    fourth_product=[[[0]*len(reps) for _ in fourth_reps] for _ in range(2)]
    for g,mask in enumerate(reps):
        mat=[[0]*order for _ in range(order)]
        for k,(i,j) in enumerate(pos):
            mat[i][j]=mat[j][i]=(mask>>k)&1
        for perm in permutations_of_order:
            objective[g]+=polynomial(mat,perm)
            k4=fourth_tab[subgraph(mat,perm[:4])]
            fourth_product[0][k4][g]+=1
            fourth_product[1][k4][g]+=mat[perm[4]][perm[5]]
            for roots in (0,2,4):
                n=(order+roots)//2
                type_=subgraph(mat,perm[:roots])
                if (roots,type_) not in by_type:
                    continue
                block,ext_ids=by_type[roots,type_]
                p=ext_ids[types_data[roots][subgraph(mat,perm[:n])]]
                q=ext_ids[types_data[roots][subgraph(mat,perm[:roots]+perm[n:])]]
                matrices[block][g][p][q]+=1
    return reps, objective, fourth_product, matrices, block_specs


def verify_occupancy():
    reps, objective, counts, flag_matrices, block_specs = flag_data()
    witness=json.loads((here/'gain6_witness.json').read_text(encoding="utf-8"))
    assert len(reps)==156
    assert len(counts[0])==11 and len(block_specs)==14
    scale=witness['scale']
    norm=factorial(order)
    last=Q('.26')
    last_bonus=Q(0)
    cases=witness['cases']
    for i, case in enumerate(cases):
        lo,hi,intercept,slope,bonus=(
            Q(case[k]) for k in ('lo','hi','intercept','slope','bonus')
        )
        assert lo==last and lo<hi and bonus>=last_bonus
        if i<len(cases)-1:
            assert intercept==Q('.018') and slope==Q('.385')
            last_bonus=bonus
        else:
            assert (lo,hi)==(Q('.46'),Q('.525'))
            assert intercept==Q('.018')+Q('.385')*Q('.446') and slope==0
        last=hi
        linear=case['linear']
        assert len(linear)==2*len(counts[0]) and min(linear)>=0
        psd=case['psd']
        assert len(psd)==len(block_specs)
        for matrix, (_,_,dimension) in zip(psd,block_specs,strict=True):
            assert len(matrix)==dimension
            check_positive_definite(matrix)
        for j,mask in enumerate(reps):
            residual=Q(objective[j],denominator*norm)
            residual-=intercept+slope*Q(mask.bit_count(),len(pairs(order)))
            for k,(p0,p1) in enumerate(zip(counts[0],counts[1],strict=True)):
                residual-=Q(linear[k],scale*norm)*(p1[j]-lo*p0[j])
                residual-=Q(linear[k+len(counts[0])],scale*norm)*(hi*p0[j]-p1[j])
            for matrix, family in zip(psd,flag_matrices,strict=True):
                residual-=Q(sum(x*y for row1,row2 in zip(matrix,family[j],strict=True)
                                    for x,y in zip(row1,row2,strict=True)),scale*norm)
            assert residual>=bonus
        print(f'Occupancy certificate verified on [{float(lo):.3f}, {float(hi):.3f}]',flush=True)
    assert Q('.725')**2>last
    assert last*(Q('0.5')-Q('.725')/6)>intercept+last_bonus


if __name__=='__main__':
    if not __debug__:
        raise SystemExit("Run without Python's -O option.")
    verify_occupancy()
