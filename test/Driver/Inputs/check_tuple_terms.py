"""The terms over tuples of tuple-terms.test, computed from the coordinates
of the dipeptide: their energies, and the forces that they add, against
central differences of those energies (kcal/mol/Å).

    check_tuple_terms.py COORDINATES PLAIN_FORCES FORCES

PLAIN_FORCES and FORCES are `mdir checkpoint --print=forces` of the runs
without and with the terms."""
import math, sys
lines=open(sys.argv[1]).read().split('\n'); n=int(lines[1].split()[0])
vals=[float(lines[2+i//6][12*(i%6):12*(i%6)+12]) for i in range(3*n)]
X=[vals[3*i:3*i+3] for i in range(n)]
def sub(a,b): return [a[k]-b[k] for k in range(3)]
def dot(a,b): return sum(a[k]*b[k] for k in range(3))
def cross(a,b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def dist(x,i,j): return math.sqrt(dot(sub(x[i],x[j]),sub(x[i],x[j])))
def angle(x,i,j,k):
    a=sub(x[i],x[j]); b=sub(x[k],x[j]); return math.acos(dot(a,b)/math.sqrt(dot(a,a)*dot(b,b)))
def dihedral(x,i,j,k,l):
    b0=sub(x[i],x[j]); b1=sub(x[k],x[j]); b2=sub(x[l],x[k])
    n1=math.sqrt(dot(b1,b1)); b1n=[c/n1 for c in b1]
    v=[b0[c]-dot(b0,b1n)*b1n[c] for c in range(3)]; w=[b2[c]-dot(b2,b1n)*b1n[c] for c in range(3)]
    return math.atan2(dot(cross(b1n,v),w), dot(v,w))
def terms(x):
    e={}
    e['harmonic']=100*(dist(x,4,6)-1.2)**2+50*(dist(x,6,8)-1.2)**2
    e['flat']=20*max(0,dist(x,1,8)-2.0)**2
    e['bend']=10*(angle(x,4,6,8)-2.0)**2
    e['twist']=1.5*(1+math.cos(2*dihedral(x,1,4,6,8)-3.0))+1.5*(1+math.cos(2*dihedral(x,4,6,8,10)-0.5))
    return e
for k,v in terms(X).items():
    print('%s %.6f' % (k, v))
# forces: -dE/dx by central differences (kcal/mol/Å)
h=1e-5; worst=0; ref=0
F={ int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[3])}
P={ int(l.split()[0]): [float(v) for v in l.split()[2:5]] for l in open(sys.argv[2])}
for a in [1,4,6,8,10]:
    for c in range(3):
        xp=[r[:] for r in X]; xm=[r[:] for r in X]; xp[a][c]+=h; xm[a][c]-=h
        g=(sum(terms(xp).values())-sum(terms(xm).values()))/(2*h)
        fnum=-g
        fm=(F[a][c]-P[a][c])/4.184/10   # kJ/mol/nm -> kcal/mol/Å
        worst=max(worst,abs(fm-fnum)); ref=max(ref,abs(fnum))
print("forces against differences of the energy: %s" % ("ok" if worst < 1e-6 * ref else "FAILED %.2e" % worst))
