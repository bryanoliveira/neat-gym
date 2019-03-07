import gym
import time


def play(net, render=False):
    env = gym.make("LunarLander-v2")
    obs = env.reset()
    score = 0

    while True:
        if render:
            env.render()
            time.sleep(0.02)

        obs = obs.tolist()
        activation = net.activate(obs)
        action = activation.index(max(activation))
        obs, reward, done, info = env.step(action)

        if done:
            break

        score += reward

    env.close()
    return score
