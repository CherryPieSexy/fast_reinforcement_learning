import gym
import crafter


def make_cart_pole():
    return gym.make('CartPole-v1')


def make_cheetah():
    return gym.make('HalfCheetah-v3')


def make_crafter():
    return gym.make('CrafterReward-v1')
