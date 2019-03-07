import gym
import time


def play(net, render=False):
    env = gym.make("BipedalWalker-v2")
    obs = env.reset()
    score = 0

    while True:
        if render:
            env.render()
            time.sleep(0.002)

        obs = obs.tolist()
        action = net.activate(obs)
        obs, reward, done, info = env.step(action)

        if done:
            break

        score += reward

    env.close()
    return score
