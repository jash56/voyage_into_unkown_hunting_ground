import csv
from pympler import tracker
from joblib import Parallel, delayed
from ai_agent import *

# Data Generation Segement
def run_agent(a):
    gridworld, dim, start, target, agent, sample = a
    movement_cost, replans_count, examination_cost = PathFinder(gridworld, np.zeros((dim, dim)), dim, start, target, agent).execute_agent(start)
    terrain = gridworld[target]
    with open("data.csv", 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([sample,dim, agent,terrain, movement_cost, examination_cost, movement_cost + examination_cost])
    return

def multi_parallel(dimension, probability, sample_size):
    arguments =[]
    for dim in range(5, dimension+1, 5):
        for sample in range(0, sample_size+1):
            gridworld, start, target = genereate_solvable_hunting_ground(dim, probability/100)
            for agent in [6, 7, 8]:
                arguments.append((gridworld, dim, start, target, agent, sample))
    Parallel(n_jobs=-1)(delayed(run_agent)(i) for i in arguments)
    print(len(arguments))

def run():
    tr = tracker.SummaryTracker()
    multi_parallel(dimension=75, probability=30, sample_size=100)
    print('end of run')
    tr.print_diff()

if __name__ == '__main__':
    run()
