import gym
import time


def play(net, render=False):
    score = 0
    env = gym.make("CartPole-v0")
    obs = env.reset()

    while True:
        if render:
            env.render()
            time.sleep(0.02)

        action = 1 if net.activate(obs)[0] > 0.5 else 0
        obs, reward, done, info = env.step(action)

        if done:
            break

        score += 1

    env.close()
    return score
