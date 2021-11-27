from collections import defaultdict
import math
import numpy as np
import heapq as hq
from copy import deepcopy

class Cell:
    def __init__(self, x, y, state=0, fval=math.inf, gval=math.inf, hval=0, parent=(-10, -10)):
        self.x = x
        self.y = y
        self.hval = hval
        self.fval = fval
        self.gval = gval
        self.parent = parent
        self.state = state

class PathFinder:
    def __init__(self, gridworld, agent_grid_view, dim, start, target, agent):
        """
        Builder function
        Takes Paramets: gridworld, agent_grid_view, belief_state, dim, start, target, agent
        """
        self.gridworld = gridworld
        self.dim = dim
        self.agent_grid_view = agent_grid_view
        self.belief_state = np.full((dim, dim), 1/(dim*dim))
        self.start = start
        self.target = target
        self.goal = (0, 0)  # Initialize Random Goal
        self.cells = {}  # Initialize Cells Hash
        self.children_hash = {}  # Initiliaze Children Hash
        self.replans_count = 0  # Counter to keep track of no of replans
        self.examination_cost = 0  # No of nodes actually examined
        self.agent = agent
        # create cell objects
        for i in range(0, dim):
            for j in range(0, dim):
                self.cells[(i, j)] = Cell(i, j)

    def heuristic(self, x1, y1):
        """
        Function used to calculate Manhattan Distance. 
        Takes row,col as input and returns the Manhattan Distance
        """
        x2, y2 = self.goal
        return abs(x1-x2) + abs(y1-y2)

    def get_children(self, cell_x, cell_y):
        """
        function used to get the next valid reachable cells from given cell
        Takes row,col as input and returns the next possible children of given cell
        """
        if self.children_hash.get((cell_x, cell_y)) != None:
            return self.children_hash.get((cell_x, cell_y))
        action_offset = [
            (1, 0),  # Right
            (0, -1),  # Down
            (-1, 0),  # Left
            (0, 1),  # Up
        ]
        children = []
        for n in action_offset:
            x = cell_x + n[0]
            y = cell_y + n[1]
            if 0 <= x < self.dim and 0 <= y < self.dim:
                children.append((x, y))
        self.children_hash[(cell_x, cell_y)] = children
        return children

    def astar(self, start_cell, goal_cell):
        """
        funciton to implement A* algorithm and caluclate path.
        Input start node and given goal node, returns path from start node to goal node and a boolean solvable to indicate if goal is reachable from the start
        """
        solvable = True
        fringe = []
        path = []
        goal = goal_cell
        start_cell = self.cells.get(start_cell)
        start_cell.fval = self.heuristic(start_cell.x, start_cell.y)
        start_cell.gval = 0
        start_cell.parent = None
        hq.heappush(fringe, (start_cell.fval, (start_cell.x, start_cell.y)))
        visited_list = {}
        while len(fringe):
            current_cell = hq.heappop(fringe)
            current_cell = current_cell[1]
            if current_cell == goal:
                path = []
                current_cell = self.cells.get(
                    (current_cell[0], current_cell[1]))
                while current_cell != None:
                    path.append((current_cell.x, current_cell.y))
                    current_cell = self.cells.get(
                        (current_cell.x, current_cell.y)).parent
                return path[::-1], True
            elif visited_list.get(current_cell) == None:
                # add cell to closed list since we are exploring
                visited_list[current_cell] = current_cell
                children = self.children_hash.get(
                    (current_cell[0], current_cell[1]), self.get_children(current_cell[0], current_cell[1]))
                # iterate through them and add them to fringe if they are not in closed set
                parent = self.cells.get((current_cell[0], current_cell[1]))
                for child_x, child_y in children:
                    if self.agent_grid_view[child_x][child_y] == 1 or visited_list.get((child_x, child_y)) != None:
                        continue
                    else:
                        child = self.cells.get((child_x, child_y))
                        child.parent = parent
                        child.gval = self.cells.get(current_cell).gval+1
                        child.hval = self.heuristic(child_x, child_y)
                        child.fval = child.gval+child.hval
                        # add to fringe lowest one of multiple entries comes out
                        hq.heappush(fringe, (child.fval, (child_x, child_y)))
        solvable = False
        return path, solvable

    def replan_path(self, cell):
        """
        Function used to replan path, when encountered a block on traversing the path given by A*
        """
        self.replans_count += 1  # counter to count number of restarts
        self.agent_grid_view[cell[0]][cell[1]] = 1
        restart_cell = self.cells.get(cell)
        if cell:
            restart_cell = self.cells.get(cell).parent
            restart_cell.parent = None
            restart_cell.gval = 0
            restart_cell.hval = self.heuristic(restart_cell.x, restart_cell.y)
            restart_cell.fval = restart_cell.hval
        return self.astar((restart_cell.x, restart_cell.y), self.goal)

    def cal_pfinding(self):
        belief_s = deepcopy(self.belief_state)
        for i in range(self.dim):
            for j in range(self.dim):
                current_fnr = self.agent_grid_view[i, j]/10
                if current_fnr == 0.0:
                    current_fnr = 0.5
                belief_s[i][j] = (1-current_fnr)*(self.belief_state[i][j])
        return belief_s

    def agent_utility(self):
        belief_s = deepcopy(self.belief_state)
        for i in range(self.dim):
            for j in range(self.dim):
                current_fnr = self.agent_grid_view[i, j]/10
                if current_fnr == 0.0:
                    current_fnr = 0.5
                belief_s[i][j] = (2-current_fnr)*(self.belief_state[i][j])/(self.heuristic(i, j)+1)
        return belief_s

    def find_goal(self):
        """
        Function to determine the next possible goal state
        Finds the most probable state from the belief state and then breaks ties based on the distance metric
        Returns the next goal state
        """
        possible_goals = defaultdict(list)  # list of possible goals
        if self.agent == 8:
            belief_s = self.agent_utility()
        elif self.agent == 7:
            belief_s = self.cal_pfinding()
        elif self.agent == 6:
            belief_s = deepcopy(self.belief_state)
        max_val = np.amax(belief_s)  # temp variable
        for row in range(self.dim):
            for col in range(self.dim):
                if(belief_s[row, col] == max_val):
                    possible_goals[row, col].append(self.heuristic(row, col))
        # find the x,y values corresponding to the cell with max proabblity and min distance
        goal = min(possible_goals.keys(), key=(lambda k: possible_goals[k]))
        return (goal)

    def examine_cell(self, cell):
        """
        Function used to examine each indermediate goal cell.
        Returns whether or not the hidden target found, and next goal state after upating the belief state
        """
        self.examination_cost += 1  # Counter used to count number of cells examined
        target_found = 0
        fnr = self.gridworld[cell[0]][cell[1]]/10  # False Negative Rate
        prior = self.belief_state[cell[0]][cell[1]]  # Prior Probability
        # If the indermediate goal state being examine is where the target is hidden then
        if cell[0] == self.target[0] and cell[1] == self.target[1]:
            # Return 1/True with probablity 1-FNR and 0/False with probability FNR
            target_found = np.random.choice([0, 1], p=[fnr, 1-fnr])
            if(target_found == 1):  # If target found then break the loop
                return target_found, cell

        # Probability of failing to find the target in cell(x,y)
        p_fail_xy = 1 - prior*(1-fnr)

        # If target not found then update the belief state
        for row in range(self.dim):
            for col in range(self.dim):
                self.belief_state[row][col] = self.belief_state[row][col]/p_fail_xy
        # updating the probabilty of the examined cell in the belief state
        self.belief_state[cell[0]][cell[1]] = prior*(fnr)/(p_fail_xy)
        # After updating the belief state, we need to recalulate the next goal i.e. max probable state closest to the current goal.

        return target_found, (self.find_goal())

    def update_blocked(self, cell):
        """
        Function to update the belief state when the agent encounters a blocked cell during execution of path given by A*
        """
        prior = self.belief_state[cell[0]][cell[1]
                                           ]  # Prior proabblity from the belief state
        # Update the belief state
        for row in range(self.dim):
            for col in range(self.dim):
                # Using probability calculate in question 2.1
                self.belief_state[row][col] = self.belief_state[row][col] / \
                    (1 - prior)
        # Setting the probability of blocked cell as zero in the belief state
        self.belief_state[cell[0]][cell[1]] = 0

        return(self.find_goal())

    def execute_agent(self, start):
        """
        Function that executes agent 6
        Takes input initial random start state
        Return the final path and cost metrics
        """
        # Initializing all values
        # print("Executing Agent", self.agent)
        start_cell = start
        self.cells.get(start_cell).parent = None
        target_found = False
        complete_path = []
        # Find path from random start state to random goal state
        path, solvable = self.astar(start_cell, self.goal)
        while not solvable:
            self.goal = (np.random.randint(self.dim),
                         np.random.randint(self.dim))
            path, solvable = self.astar(start_cell, self.goal)
        while not target_found:  # Run the loop till target not found
            for cell in path:
                # Learn about the terrain as executes the path
                self.agent_grid_view[cell[0]][cell[1]
                                              ] = self.gridworld[cell[0]][cell[1]]
                # If cell is blocked, update the belief state and replan path
                if self.agent_grid_view[cell[0]][cell[1]] == 1:
                    # Update belief state and get the new goal state
                    self.goal = self.update_blocked(cell)
                    path, solvable = self.replan_path(
                        cell)  # Replan path to goal state
                    while not solvable:
                        x = self.goal
                        self.goal = self.update_blocked(x)
                        path, solvable = self.replan_path(cell)
                    break
                # If the cell is indermediate goal state
                elif cell[0] == self.goal[0] and cell[1] == self.goal[1]:
                    start_cell = self.goal
                    target_found, self.goal = self.examine_cell(cell)
                    if target_found == 1:
                        complete_path.append(cell)
                        break
                    else:
                        path, solvable = self.astar(start_cell, self.goal)
                        while not solvable:
                            x = self.goal
                            self.goal = self.update_blocked(x)
                            path, solvable = self.astar(start_cell, self.goal)
                else:
                    complete_path.append(cell)

        if target_found:
            # print('Hidden Target found')
            return len(complete_path), self.replans_count, self.examination_cost
        else:
            print('targen not found beacuse even though exist ')


def genereate_solvable_hunting_ground(dim, probability):
    """
    Function that returns a dim x dim solvable gridworld 
    Each cell is blocked with probability p and unblocked with probability 1-p and 
    amongst the unblocked cells each terrain type is equally likely
    0 is unblocked 2 is Flat Terrain 5 is Hilly Terrain and 8 is Forest Terrain
    Takes the dimension (n) and value of prbability p as input 
    and returns a nxn gridworld along with random start state and randomly hides the target
    """
    gridworld = np.random.choice([1, 2, 5, 8], size=(dim, dim), p=[
                                 probability, 0.34*(1-probability), 0.33*(1-probability), 0.33*(1-probability)])
    start_cell_x, start_cell_y = np.random.randint(dim), np.random.randint(dim)
    target_x, target_y = np.random.randint(dim), np.random.randint(dim)
    while(gridworld[start_cell_x, start_cell_y] == 1):
        start_cell_x = np.random.randint(dim)
        start_cell_y = np.random.randint(dim)
    while(gridworld[target_x, target_y] == 1):
        target_x = np.random.randint(dim)
        target_y = np.random.randint(dim)
    path, solvable = PathFinder(gridworld, gridworld, dim, (start_cell_x, start_cell_y), (
        target_x, target_y), 6).astar((start_cell_x, start_cell_y), (target_x, target_y))
    if solvable == True:
        return gridworld, (start_cell_x, start_cell_y), (target_x, target_y)
    else:
        return genereate_solvable_hunting_ground(dim, probability/100)


# Driver Code
if __name__ == '__main__':
    dim = 10
    gridworld, start, target = genereate_solvable_hunting_ground(dim, 0.3)
    print(gridworld, "\n", "Start Cell is:",
          start, "Target is hidden at:", target)
    movement_cost, replans_count, examination_cost = PathFinder(
        gridworld, np.zeros((dim, dim)), dim, start, target, 6).execute_agent(start)
    print("Movement Cost is :", movement_cost)
    print("Examination Cost is:", examination_cost)
    print("Total Cost of Agent 6 is", movement_cost+examination_cost)
    movement_cost, replans_count, examination_cost = PathFinder(
        gridworld, np.zeros((dim, dim)), dim, start, target, 7).execute_agent(start)
    print("Movement Cost is :", movement_cost)
    print("Examination Cost is:", examination_cost)
    print("Total Cost of Agent 7 is", movement_cost+examination_cost)
    movement_cost, replans_count, examination_cost = PathFinder(
        gridworld, np.zeros((dim, dim)), dim, start, target, 8).execute_agent(start)
    print("Movement Cost is :", movement_cost)
    print("Examination Cost is:", examination_cost)
    print("Total Cost of Agent 8 is", movement_cost+examination_cost)
