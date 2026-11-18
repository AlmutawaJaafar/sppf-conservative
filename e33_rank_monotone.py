"""E33: monotonicity of the rank-p certificates in p (Section 4.5).
  example : Example 2 -- after two steps the rank-2 certificate reports a larger variance than the rank-1 certificate for
            one functional; after a change of coordinates, a larger total variance (price).
  random  : random models (n = 3..11, p = 0..5, up to 80 steps): count of steps where the total certified variance
            increases from p to p+1 (none found in 2 x 4000 models), and, in 3000 models with a tridiagonal working precision,
            count of models where C^(2) <= C^(1) fails in Loewner order.
usage: python e33_rank_monotone.py example|random
"""
import sys
import numpy as np, scipy.linalg as sl

def example():
    import numpy as np, scipy.linalg as sl
    F = np.array([[-1.5, -1.7, -2.4, -2.3], [0.0, 1.2, 1.6, -4.4], [2.7, -0.2, -0.3, 1.1], [1.3, -1.6, 1.4, 1.3]])
    J = np.diag([1.9, 0.2, 0.6, 0.2]); S = np.eye(4); Q = np.eye(4); Y = S + J; Yi = np.linalg.inv(Y)
    def rankp(A, p):
        lam, V = sl.eigh(A, Y); o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
        g = lam[p]; Z = V[:, :p] * np.sqrt(lam[:p] - g); return g * Yi + Z @ Z.T
    C = {p: Yi.copy() for p in range(4)}
    for t in range(2):
        for p in C: C[p] = rankp(S @ (F @ C[p] @ F.T + Q) @ S + J, p)
    w, V = np.linalg.eigh(C[2] - C[1]); a = V[:, -1] / np.abs(V[:, -1]).max()
    print("lambda_max(C2 - C1) =", w[-1], " functional a =", np.round(a, 3))
    print("a'C1a, a'C2a, ratio:", a @ C[1] @ a, a @ C[2] @ a, (a @ C[2] @ a) / (a @ C[1] @ a))
    a_p = np.array([1.0, 0.0, -0.3, 1.0]); print("printed functional a = (1, 0, -0.3, 1): a'C1a, a'C2a =", a_p @ C[1] @ a_p, a_p @ C[2] @ a_p)
    print("traces p=0..3:", [round(np.trace(C[p]), 4) for p in range(4)])
    print("C2 <= C0 (Cor 4):", np.linalg.eigvalsh(C[0] - C[2]).min() >= -1e-10, " C1 <= C0:", np.linalg.eigvalsh(C[0] - C[1]).min() >= -1e-10)
    
    # Change of coordinates x -> K^{1/2} x with K = a a'/|a|^2 + eps I: the transformed working filter (S~ = K^{-1/2} S K^{-1/2},
    # F~ = K^{1/2} F K^{-1/2}, Q~ = K^{1/2} Q K^{1/2}, J~ = K^{-1/2} J K^{-1/2}) is again admissible, and its certificates are
    # K^{1/2} C K^{1/2}; its total certified variance tr(K C) increases from p = 1 to p = 2.
    eps = 1e-3; K = np.outer(a, a) / (a @ a) + eps * np.eye(4); w_, U_ = np.linalg.eigh(K)
    Kh = U_ @ np.diag(np.sqrt(w_)) @ U_.T; Kih = np.linalg.inv(Kh)
    St, Ft, Qt, Jt = Kih @ S @ Kih, Kh @ F @ Kih, Kh @ Q @ Kh, Kih @ J @ Kih; Yt = St + Jt; Yti = np.linalg.inv(Yt)
    def rankp_t(A, p):
        lam, V = sl.eigh(A, Yt); o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
        g = lam[p]; Z = V[:, :p] * np.sqrt(lam[:p] - g); return g * Yti + Z @ Z.T
    Ct = {p: Yti.copy() for p in (1, 2)}
    for t in range(2):
        for p in Ct: Ct[p] = rankp_t(St @ (Ft @ Ct[p] @ Ft.T + Qt) @ St + Jt, p)
    print("transformed model: total certified variance p=1, p=2:", np.trace(Ct[1]), np.trace(Ct[2]),
          " consistent with tr(K C):", np.trace(K @ C[1]), np.trace(K @ C[2]))

def random_check():
    import subprocess
    for seed in (2, 3): subprocess.run([sys.executable, __file__, "_trace", str(seed)])
    subprocess.run([sys.executable, __file__, "_loewner"])

TRACE_SRC = 'import numpy as np, scipy.linalg as sl, sys\nrng=np.random.default_rng(int(sys.argv[1]))\ndef rankp(Y, A, p):\n    lam,V=sl.eigh(A,Y); o=np.argsort(lam)[::-1]; lam,V=lam[o],V[:,o]\n    g=max(lam[p],0.0); Z=V[:,:p]*np.sqrt(np.maximum(lam[:p]-g,0)); return g*np.linalg.inv(Y)+Z@Z.T\ndef trial(n,T,bw):\n    F=np.eye(n)*rng.uniform(0.3,1.0)+rng.uniform(0.2,1.5)*rng.standard_normal((n,n))/np.sqrt(n)\n    F*=rng.uniform(0.5,1.5)/max(abs(np.linalg.eigvals(F)))\n    Q=np.diag(rng.uniform(0.001,0.3,n)); m=int(rng.integers(1,n)); H=np.zeros((m,n)); H[np.arange(m),rng.choice(n,m,replace=False)]=1\n    J=H.T@H/rng.uniform(0.05,2)\n    P0=np.eye(n)*rng.uniform(0.01,3); Yp=np.linalg.inv(P0)\n    ps=range(0,min(n,6)); C={p:P0.copy() for p in ps}; worst=-1; arg=None\n    mask=(np.abs(np.subtract.outer(range(n),range(n)))<=bw)\n    for t in range(T):\n        Pp=F@np.linalg.inv(Yp)@F.T+Q; Pi=np.linalg.inv(Pp)\n        S=np.where(mask,Pi,0); S=(S+S.T)/2; ev=np.linalg.eigvalsh(S)\n        if ev.min()<=1e-6*ev.max(): S+=(1e-6*ev.max()-ev.min())*np.eye(n)\n        Y=S+J\n        for p in ps:\n            A=S@(F@C[p]@F.T+Q)@S+J; A=(A+A.T)/2; C[p]=rankp(Y,A,p)\n        tr=[np.trace(C[p]) for p in ps]\n        for i in range(len(tr)-1):\n            e=(tr[i+1]-tr[i])/tr[i]\n            if e>worst: worst=e; arg=(t,i)\n        Yp=Y\n    return worst,arg\nW=[]\nfor k in range(4000):\n    n=int(rng.integers(3,12)); w,a=trial(n,int(rng.integers(5,80)),int(rng.integers(0,3))); W.append(w)\n    if w>1e-9: print("VIOL",n,w,a,flush=True)\nprint("max",max(W))\n'
LOEWNER_SRC = 'import numpy as np, scipy.linalg as sl\nrng=np.random.default_rng(1)\ndef rankp(Y, A, p):\n    lam,V=sl.eigh(A,Y); o=np.argsort(lam)[::-1]; lam,V=lam[o],V[:,o]\n    g=max(lam[p],0.0); Z=V[:,:p]*np.sqrt(np.maximum(lam[:p]-g,0)); return g*np.linalg.inv(Y)+Z@Z.T\ndef trial(n,T=40):\n    F=np.eye(n)*rng.uniform(0.6,1.0)+0.4*rng.standard_normal((n,n))/np.sqrt(n)\n    F*=rng.uniform(0.7,1.3)/max(abs(np.linalg.eigvals(F)))\n    Q=np.diag(rng.uniform(0.01,0.2,n)); m=rng.integers(1,n); H=np.zeros((m,n)); H[np.arange(m),rng.choice(n,m,replace=False)]=1\n    R=np.eye(m)*rng.uniform(0.1,1); J=H.T@np.linalg.solve(R,H)\n    P0=np.eye(n)*rng.uniform(0.01,1); Yp=np.linalg.inv(P0)\n    C={1:P0.copy(),2:P0.copy(),0:P0.copy()}; worst=0; worstL=0\n    for t in range(T):\n        Pp=F@np.linalg.inv(Yp)@F.T+Q; Pi=np.linalg.inv(Pp)\n        mask=(np.abs(np.subtract.outer(range(n),range(n)))<=1)\n        S=np.where(mask,Pi,0); S=(S+S.T)/2\n        ev=np.linalg.eigvalsh(S)\n        if ev.min()<=1e-8*ev.max(): S+=(1e-8*ev.max()-ev.min())*np.eye(n)\n        Y=S+J\n        for p in C:\n            A=S@(F@C[p]@F.T+Q)@S+J; C[p]=rankp(Y,A,p)\n        worst=max(worst,(np.trace(C[2])-np.trace(C[1]))/np.trace(C[1]))\n        worstL=max(worstL,np.linalg.eigvalsh(C[2]-C[1]).max()/np.linalg.eigvalsh(C[1]).max())\n        Yp=Y\n    return worst,worstL\nres=[trial(int(rng.integers(3,9))) for _ in range(3000)]\nw=np.array(res); print("max trace excess p2 over p1",w[:,0].max(),"count>1e-9",(w[:,0]>1e-9).sum()); print("max loewner viol",w[:,1].max(),(w[:,1]>1e-9).sum())\n'

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "example"
    if mode == "example": example()
    elif mode == "random": random_check()
    elif mode == "_trace":
        sys.argv = [sys.argv[0], sys.argv[2]]
        exec(compile(TRACE_SRC, "trace", "exec"))
    elif mode == "_loewner":
        exec(compile(LOEWNER_SRC, "loewner", "exec"))
