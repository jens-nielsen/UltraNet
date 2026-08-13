import functools
import json

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch

from src.models import NeuralOperatorModel, ModelType
from src.utils import DataType, NeuralOperatorDataset, LpLoss
from src.models.chebypack import dct, idct

import src.models.chebypack as ch

x2phi_dirichlet_left = functools.partial(ch.Wrapper, [ch.dct, ch.cheb_to_shen_left])
phi2x_dirichlet_left = functools.partial(ch.Wrapper, [ch.shen_to_cheb_left, ch.idct])

def build_shen_ultraspherical_operator(a_func, N: int, num_a_modes: int = 32):
    """
    Builds the ground-truth Shen-Ultraspherical matrix operator L for:
        u'(x) + a(x) u(x) = f(x),  x in (0, 1), u(0) = 0
    
    Parameters
    ----------
    a_func : callable
        Function a(x) taking numpy arrays in [0, 1].
    N : int
        Number of spectral modes (size of output matrix N x N).
    num_a_modes : int
        Number of Chebyshev modes to resolve variable coefficient a(x).
        
    Returns
    -------
    L : np.ndarray (N x N)
        The total operator mapping Shen coefficients c -> C^(1) coefficients.
    D1 : np.ndarray (N x N)
        Derivative operator term.
    S0_Ma_P : np.ndarray (N x N)
        Variable coefficient term.
    """
    # Extended dimension to prevent intermediate truncation errors
    N_ext = N + num_a_modes + 2

    # ------------------------------------------------------------------
    # 1. Expand a(x) into Chebyshev coefficients a(y) on y in [-1, 1]
    #    Mapping: y(x) = 2x - 1  =>  x(y) = (y + 1) / 2
    # ------------------------------------------------------------------
    M = num_a_modes
    k = np.arange(M)
    y_nodes = np.cos(np.pi * (k + 0.5) / M)
    x_nodes = (y_nodes + 1.0) / 2.0
    a_vals = a_func(x_nodes)

    a_coeffs = np.zeros(M)
    a_coeffs[0] = np.mean(a_vals)
    for m in range(1, M):
        a_coeffs[m] = (2.0 / M) * np.sum(a_vals * np.cos(m * np.pi * (k + 0.5) / M))

    # ------------------------------------------------------------------
    # 2. Build D^(1): Derivative Matrix (N x N)
    #    d/dx phi_j(y) = 2 * d/dy (T_j + T_{j+1}) 
    #                  = 2j C^(1)_{j-1} + 2(j+1) C^(1)_j
    # ------------------------------------------------------------------
    D1 = np.zeros((N, N))
    for j in range(N):
        D1[j, j] = 2.0 * (j + 1)
        if j > 0:
            D1[j - 1, j] = 2.0 * j

    # ------------------------------------------------------------------
    # 3. Build P: Shen-to-Chebyshev Basis Map Matrix (N_ext x N)
    #    phi_j = T_j + T_{j+1}  =>  d_0 = c_0, d_k = c_k + c_{k-1}
    # ------------------------------------------------------------------
    P = np.zeros((N_ext, N))
    for j in range(N):
        P[j, j] = 1.0
        P[j + 1, j] = 1.0

    # ------------------------------------------------------------------
    # 4. Build M_a: Chebyshev Multiplication Operator (N_ext x N_ext)
    #    Uses T_m * T_j = 0.5 * (T_{|j-m|} + T_{j+m})
    # ------------------------------------------------------------------
    Ma = np.zeros((N_ext, N_ext))
    for j in range(N_ext):
        # m = 0 term
        Ma[j, j] += a_coeffs[0]
        # m > 0 terms
        for m in range(1, M):
            if abs(a_coeffs[m]) < 1e-15:
                continue
            val = 0.5 * a_coeffs[m]
            
            row1 = abs(j - m)
            if row1 < N_ext:
                Ma[row1, j] += val
                
            row2 = j + m
            if row2 < N_ext:
                Ma[row2, j] += val

    # ------------------------------------------------------------------
    # 5. Build S_0: Conversion Matrix C^(0) -> C^(1) (N x N_ext)
    #    T_0 = C^(1)_0
    #    T_1 = 0.5 * C^(1)_1
    #    T_j = 0.5 * (C^(1)_j - C^(1)_{j-2}) for j >= 2
    # ------------------------------------------------------------------
    S0 = np.zeros((N, N_ext))
    S0[0, 0] = 1.0
    if N > 1 and N_ext > 1:
        S0[1, 1] = 0.5
    for j in range(2, N_ext):
        if j < N:
            S0[j, j] += 0.5
        if j - 2 < N:
            S0[j - 2, j] -= 0.5

    # ------------------------------------------------------------------
    # 6. Assemble Full System Operator: L = D^(1) + S_0 * M_a * P
    # ------------------------------------------------------------------
    S0_Ma_P = S0 @ Ma @ P
    L = D1 + S0_Ma_P

    return L, D1, Ma, S0, P, S0_Ma_P


# ======================================================================
# Verification & Test Script
# ======================================================================
if __name__ == "__main__":
    # Test problem: u'(x) + (x^2 + 1) u(x) = f(x),  x in (0, 1), u(0) = 0
    poly=1
    def a_fn(x):
        left_shift = 0.5
        return 1/(1+left_shift)**poly*((x+left_shift)**poly + (x+left_shift)**(poly-1))
    N = 12

    L, D1, Ma, S0, P, S0_Ma_P = build_shen_ultraspherical_operator(a_fn, N=N, num_a_modes=64)

    print(f"=== Shen-Ultraspherical Operator L ({N}x{N}) ===")
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("\n--- Derivative Term D^(1) ---")
    print(D1)

    print("\n--- Variable Coefficient Term Ma ---")
    print(Ma)  
    
    print("\n--- Variable Coefficient Term S0 * Ma * P ---")
    print(S0_Ma_P)

    print("\n--- Full Operator Matrix L ---")
    print(L)

    print("\n--- Full Operator Inverse Matrix L-1 ---")
    print(np.linalg.inv(L))

    print("\n--- Full Operator: L-1 * L ---")
    print(np.linalg.inv(L)*L)

    # Inspect bandwidth / structure
    print(f"\nNon-zero elements: {np.count_nonzero(np.abs(L) > 1e-12)} / {N*N}")



    model_name = f"rank1sweep_{poly}"

    with open(f'./models/{model_name}.json') as f:
        d = json.load(f)

    print(d)

    data_type = DataType(d["data"])

    data = NeuralOperatorDataset(data_type, 
                                is_cheby=False if d["model"] == "fno" else True, 
                                ntrain=d["ntrain"],
                                ntest=d["ntest"], 
                                batch_size=d["batch_size"], 
                                normalize=d["normalize"],
                                subsample=d["subsample"])


    model_params = {k: eval(v) for k, v in (arg.split('=') for arg in d["arg"])}
    model = NeuralOperatorModel(ModelType(d["model"]), data=data_type, **model_params)

    loss = LpLoss(size_average=False)

    model.load_state_dict(torch.load(f"./models/{model_name}.pt", map_location=torch.device('cpu')))


    f, u = next(iter(data.test_loader))
    f_i, u_i = f[0], u[0]

    print(f[0, ..., 1])
    # 1. Flip f_i so index 0 corresponds to y = +1 (Right) for the DCT
    f_i_cgl = f_i[..., 0].flip(dims=[-1])
    f_c1 = S0[:, :N] @ np.array(dct(f_i_cgl)[:N])


    # 2. Solve system
    out = torch.zeros(f_i.shape[0])
    out[:N] = torch.tensor(np.linalg.solve(L, f_c1))

    # 3. Evaluate solution and flip back to Left-to-Right (x = 0 -> 1)
    ultra_u = phi2x_dirichlet_left(out, 0).flip(dims=[-1])

    print(loss(ultra_u[None], u_i[None]))

    import matplotlib.pyplot as plt

    plt.subplot(1, 4, 1, title="GT")
    plt.plot(f_i[..., 1], u_i)
    plt.subplot(1, 4, 2, title="USM")
    plt.plot(f_i[..., 1], ultra_u, label="USM")
    plt.subplot(1, 4, 3, title="f(x)")
    plt.plot(f_i[..., 1], f_i[..., 0], label="f(x)")
    plt.subplot(1, 4, 4, title="UM")
    plt.plot(f_i[..., 1], model(f_i[None])[0].detach().numpy(), label="UM")

    plt.show()
