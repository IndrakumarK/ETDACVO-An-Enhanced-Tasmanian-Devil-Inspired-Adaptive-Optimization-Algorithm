import torch
import math


class TDO:
    """
    Tasmanian Devil Optimization (TDO)
    Lévy-flight-based exploration.
    """

    def __init__(self,
                 dim,
                 levy_exponent=1.5,
                 step_scale=0.01,
                 device="cpu"):

        self.dim = dim
        self.lambda_ = levy_exponent
        self.step_scale = step_scale
        self.device = device

    def levy_flight(self):
        """
        Generate Lévy-distributed step using Mantegna algorithm.
        """

        beta = self.lambda_

        sigma_u = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (
                math.gamma((1 + beta) / 2)
                * beta
                * 2 ** ((beta - 1) / 2)
            )
        ) ** (1 / beta)

        u = torch.randn(self.dim, device=self.device) * sigma_u
        v = torch.randn(self.dim, device=self.device)

        step = u / torch.abs(v) ** (1 / beta)

        return step

    def update(self, theta):
        """
        theta: parameter vector (tensor shape [dim])
        """

        levy_step = self.levy_flight()

        new_theta = theta + self.step_scale * levy_step

        return new_theta