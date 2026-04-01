from __future__ import annotations

import torch


def trust_gradient_ste(x_h: torch.Tensor, xq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x_h * mask + (xq - x_h * mask).detach()


def main() -> None:
    for mask_value, expected_grad in ((1.0, 1.0), (0.0, 0.0)):
        x_h = torch.tensor([2.0], requires_grad=True)
        xq = torch.tensor([3.0])
        mask = torch.tensor([mask_value])
        y = trust_gradient_ste(x_h, xq, mask)
        if y.item() != xq.item():
            raise AssertionError(f"Expected quantized forward value {xq.item()}, got {y.item()} for mask={mask_value}")
        y.backward()
        grad = x_h.grad.item()
        if grad != expected_grad:
            raise AssertionError(f"Expected grad {expected_grad}, got {grad} for mask={mask_value}")
    print("trust_gradient_check: ok")


if __name__ == "__main__":
    main()
