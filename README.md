# Task: Implement a TCR-aware loss for Time-Optimal Continuous Authentication in PyTorch

Implement a custom PyTorch loss module called `TCRAwareLoss` for **Time-Optimal Continuous Authentication**.

## Goal

The loss should directly penalize **late impostor detection** while also penalizing **false alarms on genuine users** and optionally enforcing **temporal smoothness**.

The model outputs a sequence of scalar scores:

- `scores.shape = (B, T)`
- higher score = more likely impostor
- lower score = more likely genuine

Where:
- `B` = batch size
- `T` = sequence length

Also given:
- `labels.shape = (B,)`
  - `0` = genuine sequence
  - `1` = impostor sequence
- `attack_start.shape = (B,)`
  - integer index where attack starts for impostor sequences
  - for genuine sequences this value can be ignored

---

## Mathematical definition

Define soft detection probability at time `t`:

$$
p_t = \sigma(\alpha (s_t - \tau))
$$

Where:
- `s_t` = model score at time `t`
- `tau` = detection threshold
- `alpha` = sharpness parameter

For an impostor sequence starting at `t0`, define survival probability:

$$
S_t = \prod_{k=t_0}^{t} (1 - p_k)
$$

The differentiable surrogate for expected TCR is:

$$
L_{TCR} = \sum_{t=t_0}^{T-1} S_{t-1}
$$

with the convention that survival before the first step is `1`.

This means:
- early detection -> small loss
- delayed detection -> large loss

---

## Genuine-user usability penalty

For genuine sequences, penalize false reject probability:

$$
L_{UX} = \sum_{t=0}^{T-1} p_t
$$

This keeps genuine scores below threshold.

---

## Temporal smoothness penalty

Optional smoothness term:

$$
L_{smooth} = \sum_{t=1}^{T-1} (s_t - s_{t-1})^2
$$

This discourages unstable score oscillations.

---

## Final loss

The final loss should be:

$$
L = \lambda_{tcr} L_{TCR} + \lambda_{ux} L_{UX} + \lambda_{smooth} L_{smooth}
$$

Computed as batch averages over the relevant samples.

---

## Requirements

Implement a PyTorch module:

```python
class TCRAwareLoss(nn.Module):
    def __init__(
        self,
        tau: float = 0.0,
        alpha: float = 10.0,
        lambda_tcr: float = 1.0,
        lambda_ux: float = 1.0,
        lambda_smooth: float = 0.0,
        reduction: str = "mean",
    ):
        ...
