import neat
import os
import pickle
import importlib
import sys
import argparse
import numpy as np


# ARGS
if len(sys.argv) == 1:
    print("The folder containing a 'game.py', 'config' file and a 'genomes' folder must be an arg.")
    print("You can optionally pass the number of generations")
    exit()

parser = argparse.ArgumentParser(description='OpenAI Gym Solver')  # credits to HackerShack
parser.add_argument('--episodes', type=int, default=5,
                    help="The number of times to run a single genome. This takes the mean fitness score.")
parser.add_argument('--render', action='store_true')
parser.add_argument('--checkpoint', action='store_true')
parser.add_argument('--generations', type=int, default=50,
                    help="The number of generations to evolve the network.")
parser.add_argument('--game_name', type=str,
                    help="The folder that contains a 'game.py', a 'config' file and a 'genomes' folder.")
parser.add_argument('--num_cores', dest="num_cores", type=int, default=8,
                    help="The number cores on your computer for parallel execution.")
args = parser.parse_args()


# FUNÇÕES AUXILIARES
def save_checkpoint():
    global pop
    outdir = os.path.realpath(args.game_name + "/checkpoint")
    with open(outdir, "wb") as file:
        pickle.dump(pop, file)

    print("Saved checkpoint.")


def load_checkpoint():
    global pop
    outdir = os.path.realpath(args.game_name + "/checkpoint")
    with open(outdir, "rb") as file:
        pop = pickle.load(file)


# CONFIGS
game = importlib.import_module(args.game_name + ".game")

GENERATION = 0
RENDER_BEST = False
MAX_FITNESS = -float("inf")

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, args.game_name + "/config")

stats = neat.StatisticsReporter()

pop = None
if args.checkpoint:
    load_checkpoint()

if pop is None:
    pop = neat.Population(config)
    pop.add_reporter(stats)


def eval_genome(genome, configfile):
    global MAX_FITNESS, RENDER_BEST
    fitnesses = []
    net = neat.nn.FeedForwardNetwork.create(genome, configfile)

    for episode in range(args.episodes):
        fitness = game.play(net)
        fitnesses.append(fitness)  # game
        print("Episode: %d\tFitness: %f" % (episode, fitness))

    genome.fitness = np.array(fitnesses).mean()

    if genome.fitness > MAX_FITNESS:
        MAX_FITNESS = genome.fitness
        RENDER_BEST = True
        save_checkpoint()

    return genome.fitness


def eval_genomes(genomes, configfile):
    global GENERATION, RENDER_BEST, MAX_FITNESS
    genome_count = 0
    GENERATION += 1

    for genome_id, genome in genomes:
        genome_count += 1
        print("_" * 30)
        print("Gen: %d | Genome: %d" % (GENERATION, genome_count))
        print("-" * 10)
        eval_genome(genome, configfile)
        print("-" * 10)
        print("Mean Fitness: %d | Max Fitness: %f" % (genome.fitness, MAX_FITNESS))

        if args.render and RENDER_BEST:
            game.play(neat.nn.FeedForwardNetwork.create(genome, configfile), True)


def save(genome):
    outdir = os.path.realpath(args.game_name + "/genomes")
    number = len(os.listdir(outdir)) + 1
    file = open(outdir + "/g" + str(number) + ".gen", "wb")
    pickle.dump(genome, file)
    file.close()
    print("Saved genome g" + str(number))


if __name__ == '__main__':
    if args.render:
        winner = pop.run(eval_genomes, args.generations)
    else:
        pe = neat.parallel.ParallelEvaluator(args.num_cores, eval_genome)
        winner = pop.run(pe.evaluate, args.generations)

    print(winner)
    save(winner)
