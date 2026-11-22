"""Check the affine bounds used for the probability of no offer."""

from fractions import Fraction as Q
from math import factorial
from pathlib import Path
import json

from proof_moments import counts
from verify import check_positive_definite
from verify_gain6 import denominator, flag_data


ROOT=Path(__file__).resolve().parent
NORM=factorial(6)


def verify_file(name, pieces, triangle=0, star=0, clique=0, u='.618'):
    masks, objective, products, matrices, specs = DATA
    wit=json.loads((ROOT/f'{name}_witnesses.json').read_text(encoding="utf-8"))
    scale=wit['scale']
    assert type(scale) is int and scale>0
    pieces=[tuple(map(Q,row)) for row in pieces]
    a,b,intercept,slope=pieces[0]
    end=a
    segment=0
    theta,ss,kk,uu=map(Q,(triangle,star,clique,u))
    residual_min=None
    for case in wit['cases']:
        lo,hi,j,k=map(Q,(case['lo'],case['hi'],case['intercept'],case['slope']))
        assert end==lo and a<=lo<hi<=b and (j,k)==(intercept,slope)
        assert Q(case['theta'])==theta
        assert Q(case.get('star',0))==ss and Q(case.get('clique',0))==kk
        assert Q(case.get('u',u))==uu
        linear=case['linear']
        blocks=case['psd']
        n=len(products[0])
        assert len(linear)==2*n and all(type(v) is int and v>=0 for v in linear)
        assert len(blocks)==len(specs)
        for block,(_,_,dimension) in zip(blocks,specs,strict=True):
            assert len(block)==dimension
            assert all(type(v) is int for row in block for v in row)
            check_positive_definite(block)
        for i,mask in enumerate(masks):
            value=Q(objective[i],denominator*NORM)-intercept-slope*Q(mask.bit_count(),15)
            if theta or ss or kk:
                tri,bow,star3,mixed,cl4,cl4_mixed=counts(mask)
                value+=(theta*(bow-Q('.617')*tri)+ss*(mixed-uu*star3)
                        +kk*(cl4_mixed-uu*cl4))/NORM
            for j,(p0,p1) in enumerate(zip(*products,strict=True)):
                value-=Q(linear[j],scale*NORM)*(p1[i]-lo*p0[i])
                value-=Q(linear[n+j],scale*NORM)*(hi*p0[i]-p1[i])
            for block,table in zip(blocks,matrices,strict=True):
                total=sum(x*y for row,vector in zip(block,table[i],strict=True)
                          for x,y in zip(row,vector,strict=True))
                value-=Q(total,scale*NORM)
            assert value>=0,(name,lo,hi,mask,value)
            residual_min=value if residual_min is None else min(residual_min,value)
        end=hi
        if end==b:
            segment+=1
            if segment<len(pieces):
                a,b,intercept,slope=pieces[segment]
                assert a==end
    assert segment==len(pieces)
    print(f'{name}: {len(wit["cases"])} intervals verified, minimum residual {residual_min}',flush=True)


def verify_extensions(sharper=False):
    global DATA
    DATA=flag_data()
    verify_file('base2',[('.25','.255','.02084','.385')])
    verify_file('base',[('.255','.26','.02111','.385')])
    verify_file('tail',[
        ('.43','.53','.02835','.375'),
        ('.53','.62','.05379','.327'),
        ('.62','.70','.06681','.306'),
        ('.70','.77','.08851','.275'),
        ('.77','.89','.30026','0'),
    ])
    assert Q('.944')**2>Q('.89')
    assert Q('.89')*(Q(1,2)-Q('.944')/6)>Q('.30026')
    if sharper:
        verify_file('tail2',[
            ('.43','.50','.02750','.380'),
            ('.50','.54','.03700','.361'),
            ('.54','.60','.05644','.325'),
            ('.60','.66','.06364','.313'),
            ('.66','.70','.07156','.301'),
            ('.70','.77','.08976','.275'),
            ('.77','.89','.30151','0'),
        ])
        assert Q('.89')*(Q(1,2)-Q('.944')/6)>Q('.30151')
    verify_file('triangle',[('.255','.48','.03775','.351')],triangle='.1',u='.617')
    verify_file('star',[
        ('.25','.30','.06410','.281'),('.30','.42','.06140','.290'),
    ],star='.18',clique='.08')
    verify_file('star2',[
        ('.25','.32','.06865','.268'),('.32','.42','.06321','.285'),
    ],star='.24',clique='.025')
    verify_file('star_low',[
        ('.25','.32','.07600','.268'),('.32','.42','.07760','.263'),
    ],star='.24',clique='.025',u='.558')


if __name__=='__main__':
    if not __debug__:
        raise SystemExit('Run without Python optimization.')
    verify_extensions(sharper=True)
