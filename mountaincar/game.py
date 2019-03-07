import gym
import time


def play(net, render=False):
    env = gym.make("MountainCarContinuous-v0")
    obs = env.reset()
    score = 0
    it = 1

    while it < 200:
        if render:
            env.render()
            time.sleep(0.02)

        action = net.activate(obs)[0] * 2 - 1
        obs, reward, done, info = env.step([action])

        if done and it < 199:
            break

        score += reward
        it += 1

    env.close()
    return score
