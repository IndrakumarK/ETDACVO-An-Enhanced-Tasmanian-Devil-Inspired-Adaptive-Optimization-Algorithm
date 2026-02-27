import torch
import numpy as np


class ETDACVO:

    def __init__(self,
                 dim=9,
                 population_size=20,
                 alpha1=1.6,
                 alpha2=0.9,
                 gamma=0.2,
                 beta=0.3):

        self.dim = dim
        self.population_size = population_size
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.gamma = gamma
        self.beta = beta

        self.population = torch.randn(population_size, dim)
        self.best = None

    def decode_theta(self, theta):

        return {
            "brightness": torch.clamp(theta[0], -0.2, 0.2).item(),
            "contrast": torch.clamp(theta[1], -0.2, 0.2).item(),
            "rotation": torch.clamp(theta[2], -15, 15).item(),
            "deform_alpha": torch.clamp(theta[3], 0, 30).item(),
            "deform_sigma": torch.clamp(theta[4], 1, 5).item(),
            "noise": torch.clamp(theta[5], 0, 0.05).item(),
            "lr": torch.clamp(theta[6], 1e-5, 1e-2).item(),
            "momentum": torch.clamp(theta[7], 0.8, 0.99).item(),
            "weight_decay": torch.clamp(theta[8], 1e-6, 1e-3).item(),
        }

    def update_population(self, fitness_scores):

        fitness_scores = torch.tensor(fitness_scores)
        best_idx = torch.argmin(fitness_scores)
        self.best = self.population[best_idx].clone()

        mean = self.population.mean(dim=0)

        new_population = []

        for theta in self.population:

            levy = torch.randn(self.dim) * 0.01
            exploration = self.alpha1 * torch.rand(1) * (self.best - theta)
            diversity = self.alpha2 * torch.randn(self.dim) * (theta - mean)

            new_theta = theta + exploration + diversity + self.gamma * levy
            new_population.append(new_theta)

        self.population = torch.stack(new_population)

    def get_best(self):
        return self.best