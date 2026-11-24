import json,sys,numpy as np
def signtest_p(d):
    from math import comb
    n=int(np.sum(d!=0)); k=int(np.sum(d>0)); k=min(k,n-k)
    return min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n) if n else 1.0
def load(f): return json.load(open(f))
def g(r,k): return np.array([x[k] for x in r])
def line(name,r,ref=None):
    s=f"{name:22s} loss {g(r,'loss').mean():.4f} T {g(r,'T').mean():.3f}s stop {100*g(r,'stop').mean():.2f}cm smooth {g(r,'smooth').mean():.4f} E {g(r,'E').mean():.2f} shield {g(r,'shield').mean():.1f}% eco {g(r,'eco').mean():.1f}%"
    if ref is not None:
        d=g(r,'loss')-g(ref,'loss')
        s+=f" | dLoss {d.mean():+.4f}, worse on {np.sum(d>0)}/30 (sign p={signtest_p(d):.3f}) dE {(g(r,'E')-g(ref,'E')).mean():+.2f}"
    return s
