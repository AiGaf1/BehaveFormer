import torch


class MultiOptimizer(torch.optim.Optimizer):
    """Wraps multiple optimizers as one so Lightning's automatic optimization works."""

    def __init__(self, opts):
        self._opts = opts if isinstance(opts, list) else [opts]
        params = [p for o in self._opts for g in o.param_groups for p in g["params"]]
        lr0 = self._opts[0].param_groups[0].get("lr", 1e-3)
        super().__init__(params, {"lr": lr0})
        # Expose the inner optimizers' param_groups so LR schedulers update them
        # in place (they share dict references with the inner optimizers).
        self.param_groups = [g for o in self._opts for g in o.param_groups]

    def step(self, closure=None) -> float:  # type: ignore[override]
        loss = None
        for o in self._opts:
            result = o.step(closure)
            if result is not None:
                loss = result
        return loss or 0.0

    def zero_grad(self, set_to_none: bool = True):
        for o in self._opts:
            o.zero_grad(set_to_none=set_to_none)


def split_optimizer_params(model):
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and name.endswith("weight") and "embedding" not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


def build_muon_hybrid(model, learning_rate, config):
    muon_optimizer = getattr(torch.optim, "Muon", None)
    if muon_optimizer is None:
        raise RuntimeError("Muon optimizer requested, but this PyTorch build does not provide torch.optim.Muon.")

    muon_params, adamw_params = split_optimizer_params(model)
    optimizers = []
    if muon_params:
        optimizers.append(
            muon_optimizer(
                muon_params,
                lr=learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.muon_momentum,
                nesterov=config.muon_nesterov,
                ns_steps=config.muon_ns_steps,
                adjust_lr_fn=config.muon_adjust_lr_fn,
            )
        )
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=learning_rate, weight_decay=config.weight_decay))
    return optimizers[0] if len(optimizers) == 1 else optimizers
