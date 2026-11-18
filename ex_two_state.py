"""Two-state example (Supplement, Example S1): diagonal truncation of the prior precision gives a mean-square
bounded filter whose oracle factor is bounded, but nu > 1, so the basic (formerly "minimal") certificate grows
geometrically, while nu_sharp < 1, so the sharp certificate stays bounded (Corollary 3)."""
import numpy as np
from fractions import Fraction
F = np.array([[0.8, 1.0], [0.0, 0.8]]); Q = np.eye(2); H = np.array([[1.0, 0.0]]); R = np.array([[1.0]])
J = H.T @ H
Yi = np.eye(2); M = np.eye(2); gam = 1.0; g = []
for t in range(200):
    P = F @ Yi @ F.T + Q; S = np.diag(1 / np.diag(np.linalg.inv(P))) * 0 + np.diag(np.diag(np.linalg.inv(P)))
    Y = S + J; Yin = np.linalg.inv(Y); Ls = np.linalg.cholesky(S)
    nu = np.linalg.eigvalsh(Ls.T @ (P - Q) @ Ls).max()
    nus = np.linalg.eigvals(Yin @ S @ F @ Yi @ F.T @ S).real.max()
    gam_new = max(1, np.linalg.eigvalsh(Ls.T @ (gam * (P - Q) + Q) @ Ls).max()); g.append(gam_new / gam); gam = gam_new
    A = Yin @ S; K = Yin @ H.T @ np.linalg.inv(R); M = A @ (F @ M @ F.T + Q) @ A.T + K @ R @ K.T
    Yi = Yin
Ly = np.linalg.cholesky(Y); r = np.linalg.eigvalsh(Ly.T @ M @ Ly).max()
Pk = np.eye(2)
for t in range(500):
    Pp = F @ Pk @ F.T + Q; Pk = np.linalg.inv(np.linalg.inv(Pp) + J)
u = np.linalg.eigvalsh(Ls.T @ P @ Ls).max(); eta = 1 / np.linalg.eigvalsh(P).max()
print("steady prior P=\n", P, "\nS=diag", np.diag(S), "\nnu=", nu, " nu_sharp=", nus, " u(1-eta)=", u * (1 - eta))
print("gamma growth ratio (last)=", g[-1], " gamma_200=", gam)
print("trace M (sparse filter MSE)=", np.trace(M), " exact KF trace=", np.trace(Pk), " oracle r=", r)
print("rho(F)=", max(abs(np.linalg.eigvals(F))), " ||F||_2=", np.linalg.norm(F, 2), " rho of error map A F=", max(abs(np.linalg.eigvals(A @ F))))

# sharp certificate chain on the same working filter
import scipy.linalg as sla
Yi = np.eye(2); gs = 1.0; M = np.eye(2); hist = []
for t in range(400):
    P = F @ Yi @ F.T + Q; S = np.diag(np.diag(np.linalg.inv(P))); Y = S + J
    Pc = gs * (P - Q) + Q
    gs = sla.eigh(S @ Pc @ S + J, Y, eigvals_only=True)[-1]; hist.append(gs)
    Yin = np.linalg.inv(Y); A = Yin @ S; K = Yin @ H.T; M = A @ (F @ M @ F.T + Q) @ A.T + K @ K.T
    Yi = Yin
Ly = np.linalg.cholesky(Y); r = np.linalg.eigvalsh(Ly.T @ M @ Ly).max()
print("sharp: gamma at t=50,100,400:", hist[49], hist[99], hist[-1], " oracle r:", r, " slack:", hist[-1] / r)
