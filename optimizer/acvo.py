import torch


class ACVO:
    """
    Anti-Conservative Variable Optimization (ACVO)
    Diversity injection mechanism.
    """

    def __init__(self, dim, gamma=0.2, sigma=0.1, device="cpu"):
        """
        dim: dimensionality of parameter vector
        gamma: diversity scaling factor
        sigma: Gaussian noise std
        """
        self.dim = dim
        self.gamma = gamma
        self.sigma = sigma
        self.device = device

        # Identity correlation matrix (can be extended)
        self.C = torch.eye(dim, device=device)

    def update(self, theta):
        """
        theta: parameter vector (tensor shape: [dim])
        """

        epsilon = torch.randn(self.dim, device=self.device) * self.sigma

        perturbation = self.gamma * torch.matmul(self.C, epsilon)

        return theta + perturbation