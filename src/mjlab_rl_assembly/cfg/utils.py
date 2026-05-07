
import torch


def check_finite_tensor(name: str, x: torch.Tensor) -> torch.Tensor:
    """
    Debug helper. If tensor contains NaN/Inf, print bad env id and raise error.
    """
    if not torch.isfinite(x).all():
        bad = ~torch.isfinite(x)

        if x.ndim >= 2:
            bad_env_ids = torch.nonzero(bad.any(dim=-1), as_tuple=False).flatten()
            first_bad_env = int(bad_env_ids[0].item())
            print(f"[NaN debug] {name} has NaN/Inf.")
            print(f"[NaN debug] bad_env_ids: {bad_env_ids[:20].tolist()}")
            print(f"[NaN debug] first bad row: {x[first_bad_env]}")
        else:
            print(f"[NaN debug] {name} has NaN/Inf.")
            print(f"[NaN debug] tensor: {x}")

        raise RuntimeError(f"{name} contains NaN or Inf.")

    return x
