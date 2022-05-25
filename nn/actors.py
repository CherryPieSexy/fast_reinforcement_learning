import torch
import torch.nn as nn


class ActorCartPole(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 2)
        )

    def act(self, observation_t):
        logits = self.mlp(observation_t)
        policy = torch.distributions.Categorical(logits=logits)
        action = policy.sample()
        return action


class ActorCheetah(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(17, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.mean = nn.Linear(64, 6)
        self.log_std = nn.Linear(64, 6)

    def act(self, observation_t):
        features = self.mlp(observation_t)
        mean = self.mean(features)
        log_std = self.log_std(features)
        policy = torch.distributions.Normal(mean, log_std.exp())
        action = policy.sample()
        return action


class ActorCrafter(nn.Module):
    def __init__(self):
        super().__init__()
        # (3, 64, 64) -> (???)
        # noinspection PyTypeChecker
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2), nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256), nn.ReLU()
        )
        self.policy = nn.Linear(256, 17)

    def act(self, observation_t):
        # (b, h, w, c) -> (b, c, h, w)
        observation_t = observation_t.permute(0, 3, 1, 2)
        conv = self.conv(observation_t)
        linear = self.linear(conv.view(-1, 32 * 7 * 7))
        logits = self.policy(linear)
        policy = torch.distributions.Categorical(logits=logits)
        action = policy.sample()
        return action
