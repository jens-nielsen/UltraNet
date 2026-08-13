import torch


def shen_to_cheb_left(b: torch.Tensor) -> torch.Tensor:
    """Converts Shen basis coefficients b (for u(-1)=0) to standard Chebyshev.

    coefficients a.

    Args:
        b: Tensor of shape (..., Nx) where the last element is padded with 0,
           or shape (..., Nx-1).

    Returns:
        a: Standard Chebyshev coefficients of shape (..., Nx) satisfying
        u(-1)=0.
    """
    Nx = b.shape[-1]
    a = torch.zeros_like(b)

    # a_k = b_k + b_{k-1}
    a[..., :-1] += b[..., :-1]
    a[..., 1:] += b[..., :-1]
    return a


def cheb_to_shen_left(a: torch.Tensor) -> torch.Tensor:
    """Converts standard Chebyshev coefficients a (satisfying u(-1)=0) to Shen.

    basis coefficients b.

    Args:
        a: Chebyshev coefficients of shape (..., Nx)

    Returns:
        b: Shen coefficients of shape (..., Nx) with b[..., -1] = 0.
    """
    Nx = a.shape[-1]
    device, dtype = a.device, a.dtype

    # Generate alternating sign pattern [1, -1, 1, -1, ...]
    k = torch.arange(Nx - 1, device=device, dtype=dtype)
    sgn = torch.where(k % 2 == 0, 1.0, -1.0)

    # b_k = (-1)^k * cumsum( (-1)^j * a_j )
    b = torch.zeros_like(a)
    a_alt = a[..., :-1] * sgn
    b[..., :-1] = torch.cumsum(a_alt, dim=-1) * sgn

    return b


# -----------------------------------------------------------------------------
# Verification Script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    Nx = 8
    torch.manual_seed(42)

    # 1. Create random Shen coefficients b (last mode is 0)
    b_orig = torch.randn(2, 3, Nx)
    b_orig[..., -1] = 0.0

    # 2. Transform Shen -> Chebyshev
    a = shen_to_cheb_left(b_orig)

    # 3. Verify u(-1) = sum_k (-1)^k * a_k == 0
    k_idx = torch.arange(Nx)
    cheb_alt = torch.where(k_idx % 2 == 0, 1.0, -1.0)
    u_left_val = torch.sum(a * cheb_alt, dim=-1)
    print(
        "Max boundary error u(-1):",
        torch.max(torch.abs(u_left_val)).item(),
    )  # ~0.0

    # 4. Transform Chebyshev -> Shen (Inverse test)
    b_rec = cheb_to_shen_left(a)
    rec_error = torch.max(torch.abs(b_orig - b_rec)).item()
    print("Max roundtrip reconstruction error |b - b_rec|:", rec_error)  # ~0.0