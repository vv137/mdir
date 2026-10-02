"""The reference values of obc.mlir: the energy of generalized Born of
Onufriev, Bashford, and Case (OBC II) for five particles, from the formulas
[HawkinsCramerTruhlar, OnufrievBashfordCase2004], the forces from central
differences extrapolated in the step, and the virial, sum of x (x) F.

    python3 obc_reference.py"""
import math
X0 = [[0.09, 0.18, 0.29], [0.27, 0.2, 0.31], [0.33, 0.36, 0.29], [0.2, 0.42, 0.45], [0.41, 0.17, 0.51]]
q = [-0.5, 0.3, 0.25, -0.35, 0.3]
rho = [0.17, 0.12, 0.155, 0.15, 0.13]
S = [0.72, 0.85, 0.72, 0.79, 0.85]
off = 0.009; alpha, beta, gamma = 1.0, 0.8, 4.85
c = -(1-1/78.5)*138.935457644
n = 5
def energy(x):
    d = lambda i,j: math.dist(x[i],x[j])
    B=[]
    for i in range(n):
        ri = rho[i]-off; s = 0.0
        for j in range(n):
            if j == i: continue
            r = d(i,j); sj = S[j]*(rho[j]-off)
            if ri < r + sj:
                L = max(ri, abs(r - sj)); U = r + sj
                l = 1/L; u = 1/U
                t = l - u + 0.25*r*(u*u - l*l) + 0.5/r*math.log(u/l) + 0.25*sj*sj/r*(l*l - u*u)
                if ri < sj - r: t += 2*(1/ri - l)
                s += t
        psi = 0.5*s*ri
        B.append(1/(1/ri - math.tanh(alpha*psi - beta*psi**2 + gamma*psi**3)/rho[i]))
    e = sum(c*q[i]*q[j]/math.sqrt(d(i,j)**2 + B[i]*B[j]*math.exp(-d(i,j)**2/(4*B[i]*B[j]))) for i in range(n) for j in range(i+1,n))
    return e + sum(0.5*c*q[i]**2/B[i] for i in range(n))
def gradient(h):
    F = []
    for i in range(n):
        for k in range(3):
            xp = [r[:] for r in X0]; xm = [r[:] for r in X0]
            xp[i][k] += h; xm[i][k] -= h
            F.append(-(energy(xp) - energy(xm)) / (2 * h))
    return F
F = [(4 * b - a) / 3 for a, b in zip(gradient(1e-4), gradient(5e-5))]
W = [sum(X0[i][a] * F[3 * i + b] for i in range(n)) for a in range(3) for b in range(3)]
print('energy %.17g' % energy(X0))
print('forces', ', '.join('%.17g' % v for v in F))
print('virial', ', '.join('%.17g' % v for v in W))
