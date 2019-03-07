import pickle
import neat
import os
import importlib
import sys
import argparse

# ARGS
if len(sys.argv) == 1:
    print("The folder containing a 'game.py', 'config' file and a 'genomes' folder must be an arg.")
    print("You can optionally pass the number of generations")
    exit()

parser = argparse.ArgumentParser(description='OpenAI Gym Solver')  # credits to HackerShack
parser.add_argument('--game_name', type=str,
                    help="The folder that contains a 'game.py', a 'config' file and a 'genomes' folder.")
parser.add_argument('--episodes', type=int, default=5,
                    help="The number of times to run the simulation.")
args = parser.parse_args()

# CONFIGS

game = importlib.import_module(args.game_name + ".game")

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, args.game_name + "/config")

last = len(os.listdir(os.path.realpath(args.game_name + "/genomes")))
if last == 0:
    print("Train your network first!")
    exit()

genomeFile = os.path.realpath(args.game_name + "/genomes/g" + str(last) + ".gen")
genome = pickle.load(open(genomeFile, "rb"))

print("Running genome g" + str(last))
for i in range(args.episodes):
    fitness = game.play(neat.nn.FeedForwardNetwork.create(genome, config), True)
    print("Fitness is %f" % fitness)
