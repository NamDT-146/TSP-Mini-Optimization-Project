'''
Input:
    - N: Number of nodes (including depot)
    - t[i][j]: Travel time between node i and node j
    - e[i], l[i]: Earliest and latest allowable times for node i
    - d[i]: Service time at node i
    - K: Number of ants
    - MaxIter: Maximum number of iterations
    - Parameters: α (pheromone influence), β (heuristic influence), ρ (evaporation rate), Q (pheromone deposit factor)

Output:
    - BestSolution: Best path found
    - BestCost: Cost of the best solution

Initialize:
    - Initialize pheromone matrix τ[i][j] = τ₀ for all i, j
    - Define heuristic information η[i][j] = 1 / t[i][j]
    - BestSolution ← ∅, BestCost ← ∞

For iter = 1 to MaxIter do:
    Solutions ← ∅

    For each ant k = 1 to K do:
        - Initialize CurrentNode ← Depot (node 0)
        - Initialize Path ← [Depot]
        - Initialize ArrivalTime ← t₀ (start time)
        - Initialize Visited ← {Depot}

        While |Path| < N + 1 do:
            - Calculate Probabilities for moving to each unvisited node j:
                If j ∉ Visited and ArrivalTime + t[CurrentNode][j] satisfies time windows:
                    P[j] = (τ[CurrentNode][j]^α) * (η[CurrentNode][j]^β)
                Else:
                    P[j] = 0

            - Normalize P[j] to make it a probability distribution.
            - Select next node j based on P[j] (roulette-wheel selection).
            - Update Path, Visited, and ArrivalTime:
                ArrivalTime = max(e[j], ArrivalTime + t[CurrentNode][j]) + d[j]
                Append j to Path, mark j as visited.

        - Calculate Cost for the ant's Path.
        - If Path violates constraints (e.g., time windows):
            Assign a high penalty cost to the Path.
        - Add Path and its Cost to Solutions.

    Update Pheromone:
        - Evaporate pheromones: τ[i][j] = (1 - ρ) * τ[i][j] for all i, j
        - For each ant solution in Solutions:
            If solution is feasible:
                Deposit pheromones along the path:
                τ[i][j] += Q / Cost

    Update BestSolution and BestCost:
        - If any solution in Solutions is better than BestCost:
            Update BestSolution and BestCost.

Return BestSolution, BestCost

'''

import random, math
from utils import read_input, evaluate
from Greedy import greedy_tsp_with_time_windows

class ACO_Solver():
    # Global variables for pheromone and heuristic matrices

    def __init__(self, num_ants, iteration, alpha, beta, theta, evaporation_rate, N, time_windows, travel_time):
        self.N = N
        self.time_windows = time_windows
        self.travel_time = travel_time

        self.pheromone = []
        self.heuristic = []
        self.time_heuristic = []
        
        self.initialize_pheromone_matrix(N)
        self.construct_heuristic()
        self.construct_time_heuristic()
        # print(time_heuristic)
        # print(heuristic)

        greedy_solution, cost = greedy_tsp_with_time_windows(N, travel_time, time_windows)
        print(cost)

        prev_node = 0
        for next_node in greedy_solution:
            self.pheromone[prev_node][next_node] = 0.9999
            prev_node = next_node

        self.alpha = alpha # Pheromone influence
        self.beta = beta  # Heuristic influence
        self.theta = theta
        self.evaporation_rate = evaporation_rate
        self.Q = 1000  # Pheromone deposit constant
        self.num_ants = num_ants
        self.num_iterations = iteration

    def initialize_pheromone_matrix(self, N, initial_pheromone=0.05):
        """
        Initialize the pheromone matrix with a given initial pheromone level.
        
        Args:
            n (int): Number of nodes.
            initial_pheromone (float): Initial pheromone level for all edges.

        Returns:
            list of lists: Pheromone matrix.
        """
        self.pheromone = [[initial_pheromone for _ in range(N + 1)] for _ in range(N + 1)]

    def construct_heuristic(self):
        """
        Construct the heuristic matrix based on travel time.

        Args:
            travel_time (list of lists): Matrix of travel times between nodes.

        Returns:
            list of lists: Heuristic matrix where heuristic[i][j] is the inverse of travel_time[i][j].
        """
        max_time_travel = 0
        for row in self.travel_time:
            max_time_travel = max(max_time_travel, max(row))

        self.heuristic = [[  max_time_travel / (self.travel_time[i][j]) if self.travel_time[i][j] > 0 else 0 for j in range(self.N + 1)] for i in range(self.N + 1)]
        mx_ele = 0
        mn_ele = float('inf')
        for row in self.heuristic:
            mx_ele = max(mx_ele, max(row))
            mn_ele = min(mn_ele, min(row))
        
        self.heuristic = [[ (e - mn_ele)/mx_ele for e in row ] for row in self.heuristic ]

    def construct_time_heuristic(self):
        """
        Construct the heuristic matrix based on travel time.

        Args:
            travel_time (list of lists): Matrix of travel times between nodes.

        Returns:
            list of lists: Heuristic matrix where heuristic[i][j] is the inverse of travel_time[i][j].
        """
    
        l = [ l for (e, l, d) in self.time_windows]
        self.time_heuristic = [0.99] + [ 0.99 for (e, l, d) in self.time_windows[1:]]
        mx_ele = max(self.time_heuristic)
        mn_ele = min(self.time_heuristic)
        if (mx_ele != mn_ele):
            self.time_heuristic = [(e - mn_ele) / mx_ele for e in self.time_heuristic]

    def update_pheromone(self, solutions):
        """
        Update the pheromone levels based on the solutions found by ants.

        Args:
            solutions (list of tuples): List of (solution, cost) pairs.
            evaporation_rate (float): Rate at which pheromone evaporates.
            Q (float): Constant for pheromone deposit.

        Returns:
            None
        """     
        max_pheromone = 0.9999
        min_pheromone = 0.0001
        # Evaporate pheromone
        for i in range(self.N + 1):
            for j in range(self.N + 1):
                self.pheromone[i][j] *= (1 - self.evaporation_rate)

        # Deposit pheromone
        for solution, cost in solutions:
            # print("cf")
            if cost == -1:
                for i in range( len(solution) ):
                    self.time_heuristic[solution[i]] = ( self.time_heuristic[solution[i]] / (self.N + 1)**1.5 * (self.N - i)**1.5 ) ** (3/4)
                mx_ele = max(self.time_heuristic)
                mn_ele = min(self.time_heuristic)
                if (mx_ele != mn_ele):
                    self.time_heuristic = [(e - mn_ele) / mx_ele for e in self.time_heuristic]
                pheromone_delta = 0
                self.time_heuristic = [min( max_pheromone, max(min_pheromone, i) ) for i in self.time_heuristic]

            else:
                pheromone_delta = self.Q / cost
                for k in range(len(solution) - 1):
                    i, j = solution[k], solution[k + 1]
                    self.pheromone[i][j] += pheromone_delta # One direction 
                    self.pheromone[i][j] = min( max_pheromone, max(min_pheromone, self.pheromone[i][j]) )
                # pheromone[i][j] = max(0, pheromone[i][j])


    def choose_next_node(self, current_node, visited, timer):
        """
        Choose the next node based on pheromone and heuristic information.

        Args:
            current_node (int): Current position of the ant.
            visited (set): Set of visited nodes.
            alpha (float): Pheromone influence.
            beta (float): Heuristic influence.

        Returns:
            int: The chosen next node.
        """
        probabilities = []
        total_prob = 0
        for next_node in range(1, self.N + 1):
            start_time, end_time, dur = self.time_windows[next_node]
            if next_node not in visited and (timer + self.travel_time[current_node][next_node] <= end_time):
                prob = (self.pheromone[current_node][next_node] ** self.alpha) * (self.heuristic[current_node][next_node] ** self.beta) * (self.time_heuristic[next_node] ** self.theta)
                probabilities.append((next_node, prob))
                total_prob += prob

        if total_prob == 0:
            return -1, -1

        # Apply exponential weighting
        # probabilities = [(next_node, math.exp(prob)) for next_node, prob in probabilities]
        
        # # Normalize probabilities
        total = sum(prob for _, prob in probabilities)
        probabilities = [(next_node, prob / total) for next_node, prob in probabilities]
        # print(probabilities)

        r = random.random()
        cumulative = 0
        for next_node, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                start_time, end_time, dur = self.time_windows[next_node]
                timer = max(timer + self.travel_time[current_node][next_node], start_time) + dur
                return timer, next_node


    def construct_solution(self):
        """
        Construct a solution for an ant.

        Args:
            n (int): Number of nodes.
            pheromone (list of lists): Pheromone matrix.
            heuristic (list of lists): Heuristic matrix (e.g., inverse of distance).
            alpha (float): Pheromone influence.
            beta (float): Heuristic influence.

        Returns:
            list: A solution represented as a sequence of node indices.
        """
        solution = []
        visited = set()
        current_node = 0
        timer = 0


        while len(visited) < self.N:
            timer, next_node = self.choose_next_node(current_node, visited, timer)
            if timer == -1:
                return False, solution, timer
            solution.append(next_node)
            visited.add(next_node)
            current_node = next_node

            # print(solution)
        
        timer += self.travel_time[current_node][0]

        return True, solution, timer

    def Solve(self):
        
        best_solution = None
        best_cost = float('inf')

        for iteration in range(self.num_iterations):
            solutions = []
            for _ in range(self.num_ants):
                is_valid, solution, cost = self.construct_solution()
                
                solutions.append((solution, cost))

                if is_valid:
                    if cost < best_cost:
                        best_solution = solution
                        best_cost = cost
                

            self.update_pheromone(solutions)
            # for row in pheromone:
            #     for e in row:
            #         print(f"%.2f"%e, end=" ")
            #     print()
            print(f"Iteration {iteration + 1}: Best Cost = {best_cost}")

        # print("Best Solution:", best_solution)
        # print("Best Cost:", best_cost)
        return(best_solution, best_cost)

# Example usage
if __name__ == "__main__":
    use_file = True
    file_path = "TestCase\Subtask_100\\rc208.2"
    
    N, time_windows, travel_time = read_input(from_file=use_file, file_path=file_path)

    Solver = ACO_Solver(num_ants = 400, iteration = 1000, alpha = 2.0, beta = 0, theta = 1.0, evaporation_rate = 0.1, N = N, time_windows = time_windows, travel_time = travel_time)
    best_solution, best_cost = Solver.Solve()


    print("Best Solution:", best_solution)
    print("Best Cost:", best_cost)
