"""Opt-in CUDA float64 objective for native boundary-distribution fitting.

SciPy still performs the identical optimization with the same initializations,
bounds, fold membership and stopping settings. Only the objective/gradient
matrix operations move. It is a numerically tested alternative, NOT a claim
of bit-identical optimizer trajectories. MPS fitting stays on CPU because its
float32-only objective can change a tightly converged model; MPS predictive
scoring is separately supported with exact CPU boundary rechecks.
"""
from __future__ import annotations

import numpy as np


class CudaObjective:
    def __init__(self, xy, area, scaled, offset, exposure, *, maximum_bytes):
        import torch
        self.torch = torch
        required = 8*sum(v.size for v in (xy, area, scaled, offset, exposure))
        if required*3 > maximum_bytes:
            raise MemoryError('CUDA fit working set exceeds accelerator budget')
        tensor = lambda a: torch.as_tensor(np.array(a, copy=True), device='cuda', dtype=torch.float64)
        self.xy, self.area, self.scaled, self.offset, self.exposure = map(tensor, (xy, area, scaled, offset, exposure))

    def __call__(self, parameters):
        t = self.torch
        with t.inference_mode():
            p = t.as_tensor(parameters, device='cuda', dtype=t.float64)
            a, b, c = p[2].exp(), p[3], p[4].exp()
            dx = self.xy-p[:2]
            z = a*dx[:, 0]+b*dx[:, 1]; w = c*dx[:, 1]
            density = -.5*(z*z+w*w)
            derivative = t.stack((a*z, b*z+c*w, -z*a*dx[:, 0], -z*dx[:, 1], -w*w), dim=1)
            q = t.softmax(density+self.area, dim=0)
            joint = self.scaled @ q; prior = self.exposure @ q
            # Stable CPU fallback is selected by the caller on invalid or
            # underflowed rows; do not clip or change the likelihood model.
            if bool(((joint < 1e-180) | (prior < 1e-180)).any().item()):
                return None
            weights = q*(self.scaled.T @ (1./joint)-self.exposure.T @ (1./prior))
            objective = -(joint.log()-prior.log()+self.offset).sum()
            gradient = -(weights @ derivative)
            result = t.cat((objective[None], gradient)).cpu().numpy()
            return float(result[0]), result[1:]
