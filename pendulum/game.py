import gym
import time


def play(net, render=False):
    env = gym.make("Pendulum-v0")
    obs = env.reset()
    score = 0

    while True:
        if render:
            env.render()
            time.sleep(0.02)

        obs = obs.tolist()
        action = net.activate(obs)[0] * 4 - 2
        obs, reward, done, info = env.step([action])

        if done:
            break

        score += reward

    env.close()
    return score
