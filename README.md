# voyage_into_unkown_hunting_ground
# AI Project
Designed a blindfolded AI agent using Bayesian Networks for optimally searching a hidden target within a simulated grid world environment

We explore gridworlds but of a different type. We have a dim * dim square grid, where cells are blocked with probability p=0.3. 
The unblocked cells have an equal likelihood of having flat, hilly, or forest terrains. We have represented the gridworld using a NumPy array of 1’s, 2’s, 5’s, and 8’s. 
A 1 stands for blocked cell, 2 stands for unblocked cell with flat terrain, 5 stands for unblocked cell with hilly terrain, and 8 stands for unblocked cell with forest terrain. 
## Implementation:
A robot starts from a random position in this gridworld and attempts to find the hidden target. We run A* on the gridworld to check the solvability of our randomly generated gridworld. If the target is not reachable from this start state, we drop that case and start until we find a solvable combination.
Every robot maintains its separate agent gridview, a NumPy array of 0’s, 1’s, 2’s, 5’s, and 8’s. Initially, the robot assumes the entire gridworld to be unblocked(all zeroes) and updates the agent grid-view as and when it traverses through the gridworld. Our robots are blindfolded (can tell a cell is blocked when it attempts to move there and fails). When the robot moves into an unblocked cell, it can sense the terrain type. Based on the block cells encountered and the terrain it senses while moving around the grid, the robot will keep updating its agent gridview. 
The robot can move between unblocked cells in 4 cardinal directions. Our robot also maintains a children_hash which is populated at the zeroth stage. children_hash is used for getting the 4 possible cells the robot can move into next in our A* algorithm.
Our robot can examine the cell it is currently in to try and find the target. But there could be false negatives. If the target is in the cell we are examining,, the examination fails with different probabilities based on the terrain. The false-negative rate is 0.2 for flat terrain, 0.5 for hilly, and 0.8 for forests. If the robot finds the target in a cell, the gridworld is solved as there are no false-positive rates.
Our robot maintains a belief state of the entire gridworld. This belief state is a NumPy array of size dim*dim, which stores the probability of the target being in each cell. These probabilities are updated based on the information our robot gains from the gridworld. All three agents in this project maintain the same belief state i.e., maintain the probability of each cell in the grid containing the target. But the decision of where to go/what to do next is different for each of them.