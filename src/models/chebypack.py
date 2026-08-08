import torch
import numpy as np


def dct(u):
    Nx = u.shape[-1]

    # transform x -> theta, a discrete cosine transform of "cheap" version
    V = torch.cat([u, u.flip(dims=[-1])[..., 1 : Nx - 1]], dim=-1)
    a = torch.fft.ifft(V, dim=-1)[..., :Nx].real
    a[..., 1 : Nx - 1] *= 2
    return a

def idct(a):
    Nx = a.shape[-1]

    v = a.clone()
    v[..., (0, Nx - 1)] *= 2
    V = torch.cat([v, v.flip(dims=[-1])[..., 1 : Nx - 1]], dim=-1)
    u = torch.fft.fft(V, dim=-1)[..., :Nx].real / 2
    return u


def cmp(a):
    Nx = a.shape[-1]

    sgn = torch.zeros_like(a)
    sgn[..., ::2] = 1.0
    b = torch.fft.irfft(
        torch.fft.rfft(sgn, n=2 * Nx, dim=-1) * torch.fft.rfft(a, n=2 * Nx, dim=-1),
        dim=-1,
    )[..., :Nx]
    # b[..., -4:-2] = -a[..., -2:]
    # print(b[0, 0, ..., -2:])
    # b[..., -2:] = 0
    return b


def cmp_decrease(a):
    Nx = a.shape[-1]

    sgn = torch.zeros(*a.shape[:-1], 2 * Nx, dtype=a.dtype, device=a.device)
    sgn[..., -(Nx - 1) // 2 * 2 :: 2] = -1.0

    b = torch.fft.irfft(
        torch.fft.rfft(sgn, n=2 * Nx, dim=-1) * torch.fft.rfft(a, n=2 * Nx, dim=-1),
        dim=-1,
    )[..., :Nx]
    b[..., -2:] = 0
    # res = b[..., :2] - a[..., :2]
    # b[..., :2] = a[..., :2]

    return b


def cmp_neumann(a):
    Nx = a.shape[-1]
    fac = torch.linspace(0, Nx - 1, Nx, dtype=a.dtype, device=a.device) ** 2

    sgn = torch.zeros_like(a)
    sgn[..., ::2] = 1

    b = torch.fft.irfft(
        torch.fft.rfft(sgn, n=2 * Nx, dim=-1)
        * torch.fft.rfft(a * fac, n=2 * Nx, dim=-1),
        dim=-1,
    )[..., :Nx]

    b[..., :2] = a[..., :2]
    b[..., 2:-2] /= fac[2:-2]
    # b[..., -2:] = 0
    # b[..., -4:-2] = -a[..., -2:] / torch.tensor(\
    #     [(Nx-4.0)/(Nx-2.0),(Nx-3.0)/(Nx-1.0)], dtype=torch.float64, device=a.device)**2

    return b


def cmp_robin0(a):
    Nx = a.shape[-1]
    fac = torch.linspace(0, Nx - 1, Nx, dtype=a.dtype, device=a.device)
    fac = (fac - 1.0) * (fac + 1.0)

    sgn = torch.zeros_like(a)
    sgn[..., ::2] = 1

    b = torch.fft.irfft(
        torch.fft.rfft(sgn, n=2 * Nx, dim=-1)
        * torch.fft.rfft(a * fac, n=2 * Nx, dim=-1),
        dim=-1,
    )[..., :Nx]

    b[..., :2] = a[..., :2]
    b[..., 2:-2] /= fac[2:-2]
    # b[..., -2:] = 0
    # b[..., -4:-2] = -a[..., -2:] / torch.tensor(\
    #     [(Nx-4.0)/(Nx-2.0),(Nx-3.0)/(Nx-1.0)], dtype=torch.float64, device=a.device)**2
    return b


def cmp_robin(a):
    Nx = a.shape[-1]
    fac = torch.linspace(0, Nx - 1, Nx, dtype=a.dtype, device=a.device)
    fac = fac**2 + 1
    # fac = (fac-1.0)*(fac+1.0)

    sgn = torch.zeros_like(a)
    sgn[..., ::2] = 1

    b = torch.fft.irfft(
        torch.fft.rfft(sgn, n=2 * Nx, dim=-1)
        * torch.fft.rfft(a * fac, n=2 * Nx, dim=-1),
        dim=-1,
    )[..., :Nx]

    b[..., :2] = a[..., :2]
    b[..., 2:-2] /= fac[2:-2]
    return b


def icmp(b):
    Nx = b.shape[-1]
    a = torch.zeros_like(b)
    a[..., :2] = b[..., :2]
    a[..., 2:] = b[..., 2:] - b[..., : Nx - 2]
    a[..., -2:] = -b[..., Nx - 4 : Nx - 2]

    return a


def icmp_neumann(b):
    Nx = b.shape[-1]
    a = torch.zeros_like(b)

    p = torch.linspace(0, Nx - 3, Nx - 2, dtype=torch.float32, device=b.device)
    p = (p / (p + 2.0)) ** 2
    a[..., 0:2] = b[..., 0:2]
    a[..., 2 : Nx - 2] = b[..., 2 : Nx - 2] - p[: Nx - 4] * b[..., : Nx - 4]
    a[..., Nx - 2 : Nx] = -p[Nx - 4 : Nx - 2] * b[..., Nx - 4 : Nx - 2]

    return a


def icmp_robin0(b):
    Nx = b.shape[-1]
    a = torch.zeros_like(b)

    p = torch.linspace(0, Nx - 3, Nx - 2, dtype=torch.float32, device=b.device)
    p = (p - 1.0) / (p + 3.0)

    a[..., 0:2] = b[..., 0:2]
    a[..., 2 : Nx - 2] = b[..., 2 : Nx - 2] - p[: Nx - 4] * b[..., : Nx - 4]
    a[..., Nx - 2 : Nx] = -p[Nx - 4 : Nx - 2] * b[..., Nx - 4 : Nx - 2]

    return a


def icmp_robin(b):
    # pk = (p * k^2 +1) / (p * (k+2)^2 + 1)
    Nx = b.shape[-1]
    a = torch.zeros_like(b)

    pk = torch.linspace(0, Nx - 3, Nx - 2, dtype=torch.float32, device=b.device)
    pk = (pk**2 + 1) / ((pk + 2.0) ** 2 + 1)

    a[..., 0:2] = b[..., 0:2]
    a[..., 2 : Nx - 2] = b[..., 2 : Nx - 2] - pk[: Nx - 4] * b[..., : Nx - 4]
    a[..., Nx - 2 : Nx] = -pk[Nx - 4 : Nx - 2] * b[..., Nx - 4 : Nx - 2]

    return a


def Wrapper(func_list, u, dim):
    # a wrapper to apply a list of function on given axises.
    # the func will be applied in turn.
    if type(dim) == int:
        dim = [dim]
    total_dim = u.dim()

    for d in dim:
        if (d != total_dim - 1) and (d != -1):
            u = torch.transpose(u, d, -1)

        for func in func_list:
            u = func(u)

        if (d != total_dim - 1) and (d != -1):
            u = torch.transpose(u, d, -1)
    return u


"""
def cheb_partial(u, d):
    Nx, total_dim = u.shape[d], u.dim()
    if d != total_dim-1:
        u = torch.transpose(u, d, total_dim-1)

    tmp = torch.cat([u, u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)

    a = torch.fft.ifft(tmp, dim=-1) * 2
    a = torch.real(a[..., :Nx])
    a[..., 0] /= 2; a[..., Nx-1] /= 2

    a = a[..., 1:] # make sure that N=2^k for FFT

    a *= 2 * torch.linspace(1, Nx-1, Nx-1, dtype=torch.float64, device=u.device)

    a = torch.flip(a, [-1])
    sgn = torch.zeros_like(a, device=u.device)
    sgn[..., 1::2] = 1

    b = torch.fft.irfft(torch.fft.rfft(sgn, n=2*(Nx-1), dim=-1)
                        * torch.fft.rfft(a, n=2*(Nx-1), dim=-1), dim=-1)

    b = torch.flip(b[..., :Nx], [-1])
    b[..., 0] /= 2
    #b[..., Nx-1] = 0

    a = b

    a[..., 0] *= 2; a[..., Nx-1] *= 2

    tmp = torch.cat([a, a.flip(dims=[-1])[..., 1:Nx - 1]], dim=-1)
    #tmp = np.concatenate([a, np.flip(a, axis=[-1])[..., 1:Nx-1]], axis=-1)
    u = torch.fft.fft(tmp, dim=-1) / 2
    u = torch.real(u[..., :Nx])

    u = torch.transpose(u, d, total_dim-1)
    return u
"""


def cheb_partial(u, d, truc=None):
    Nx, total_dim = u.shape[d], u.dim()
    if d != total_dim - 1 and d != -1:
        u = torch.transpose(u, d, total_dim - 1)

    V = torch.cat([u, u.flip(dims=[-1])[..., 1 : Nx - 1]], dim=-1)
    a = torch.fft.ifft(V, dim=-1)[..., :Nx].real
    a[..., 1 : Nx - 1] *= 2

    a *= 2 * torch.linspace(0, Nx - 1, Nx, dtype=torch.float32, device=u.device)
    sgn = torch.zeros(2 * Nx, device=a.device, dtype=torch.float32)
    sgn[..., Nx // 2 * 2 + 1 :: 2] = 1

    b = torch.fft.irfft(
        torch.fft.rfft(sgn, n=2 * Nx, dim=-1) * torch.fft.rfft(a, n=2 * Nx, dim=-1),
        dim=-1,
    )[..., :Nx]

    if truc != None:
        b[..., truc:] = 0

    b[..., 0] /= 2
    # b[..., Nx-1] = 0

    a = b

    a[..., 1 : Nx - 1] /= 2
    V = torch.cat([a, a.flip(dims=[-1])[..., 1 : Nx - 1]], dim=-1)
    u = torch.fft.fft(V, dim=-1)[..., :Nx].real  # / 2

    if d != total_dim - 1 and d != -1:
        u = torch.transpose(u, d, total_dim - 1)
    return u


Dx = cheb_partial


def cmp_UpperDirichlet(a):
    b = a.cumsum(dim=-1)
    b[..., -2] = -a[..., -1]
    b[..., -1] = 0
    return b


def icmp_UpperDirichlet(b):
    a = torch.zeros_like(b)
    a[..., 1:-1] = b[..., 1:-1] - b[..., :-2]
    a[..., 0] = b[..., 0]
    a[..., -1] = -b[..., -2]
    return a
import numpy as np

def cmp_robin_v(a, S_cheb_to_comp):
    modes = S_cheb_to_comp.shape[0]
    a_cmp = torch.einsum('...i,ji->...j', a[..., :modes], S_cheb_to_comp)
    return torch.cat((a_cmp, a[..., modes:]), dim=-1)


def icmp_robin_v(a, S_comp_to_cheb):
    modes = S_comp_to_cheb.shape[0]
    a_icmp =  torch.einsum('...i,ji->...j', a[..., :modes], S_comp_to_cheb)
    return torch.cat((a_icmp, a[..., modes:]), dim=-1)


def get_square_robin_transforms(N, a_L, b_L, a_R, b_R):
    """
    Computes the square NxN transformation matrices between the Chebyshev basis 
    and the augmented compact basis.
    """
    n = np.arange(N)
    
    L = a_L * ((-1)**n) + b_L * ((-1)**(n+1) * n**2)
    R = a_R * np.ones(N) + b_R * (n**2)
    
    # Initialize the square NxN matrix
    S_comp_to_cheb = np.zeros((N, N))
    
    # 1. Fill the first N-2 columns (The homogeneous compact basis)
    for k in range(N - 2):
        L_k, L_kp1, L_kp2 = L[k], L[k+1], L[k+2]
        R_k, R_kp1, R_kp2 = R[k], R[k+1], R[k+2]
        
        D_k = L_kp1 * R_kp2 - L_kp2 * R_kp1
        
        if np.abs(D_k) < 1e-14:
            raise ValueError(f"Degenerate boundary condition for mode k={k}.")
            
        alpha_k = (-L_k * R_kp2 + L_kp2 * R_k) / D_k
        beta_k  = (-L_kp1 * R_k + L_k * R_kp1) / D_k
        
        S_comp_to_cheb[k, k]     = 1.0
        S_comp_to_cheb[k+1, k]   = alpha_k
        S_comp_to_cheb[k+2, k]   = beta_k
        
    # 2. Fill the last 2 columns (The boundary lifters)
    S_comp_to_cheb[N-2, N-2] = 1.0
    S_comp_to_cheb[N-1, N-1] = 1.0
    
    # 3. Invert the matrix (exact inversion since it's perfectly lower triangular)
    S_cheb_to_comp = np.linalg.inv(S_comp_to_cheb)
    
    return torch.tensor(S_comp_to_cheb, dtype=torch.float32), torch.tensor(S_cheb_to_comp, dtype=torch.float32)

def test_square_basis_transforms():
    N = 16
    a_L, b_L = 1.5, -0.5
    a_R, b_R = 2.0, 1.2
    
    S_forward, S_inverse = get_square_robin_transforms(N, a_L, b_L, a_R, b_R)
    
    # --- Test A: Lossless transformation of an ARBITRARY field ---
    np.random.seed(42)
    u_hat_arbitrary = np.random.randn(N)
    
    # Map from Chebyshev to Compact
    c_augmented = S_inverse @ u_hat_arbitrary
    
    # Map back from Compact to Chebyshev
    u_hat_recovered = S_forward @ c_augmented
    
    error_arbitrary = np.max(np.abs(u_hat_arbitrary - u_hat_recovered))
    print(f"Test A (Arbitrary Field) - Recovery Error: {error_arbitrary:.2e}")
    print(f"Notice the last two compact coeffs are non-zero: "
          f"[{c_augmented[-2]:.2f}, {c_augmented[-1]:.2f}]\n")
    
    # --- Test B: Transformation of a field satisfying HOMOGENEOUS BCs ---
    # We create one by generating random N-2 coeffs and mapping forward
    c_homogeneous = np.zeros(N)
    c_homogeneous[:-2] = np.random.randn(N - 2)
    
    # This u_hat perfectly satisfies the homogeneous boundary conditions
    u_hat_homo = S_forward @ c_homogeneous 
    
    # Map back to compact coefficients
    c_recovered = S_inverse @ u_hat_homo
    
    error_homo = np.max(np.abs(c_homogeneous - c_recovered))
    print(f"Test B (Homogeneous Field) - Recovery Error: {error_homo:.2e}")
    print(f"Notice the last two compact coeffs are exactly zero: "
          f"[{c_recovered[-2]:.2e}, {c_recovered[-1]:.2e}]")

if __name__ == "__main__":
    test_square_basis_transforms()
