import random
from utils import evaluate, read_input
from Greedy import greedy_tsp_with_time_windows

class GASolves():
        
    def __init__(self, N, time_windows, travel_time, population_size, generations, mutation_rate, tournament_size, elitism_size):
        self.N = N
        self.population_size = population_size
        self.time_windows = time_windows
        self.travel_time = travel_time
        self.population = []
        self.initialize_population()
        self.fitness_value = []
        print(self.population)
        self.update_fitness_value()
        self.penalty_factor = 0.5
        # print(self.fitness_value)
        self.generations = generations
        self.mutation_rate = mutation_rate 
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size

    def add_random_individuals(self, num_random_individuals):
        """ Add random individuals to the population to increase diversity. """
        for _ in range(num_random_individuals):
            random_solution = random.sample(range(1, self.N + 1), k=self.N)
            self.population.append(random_solution)
    
    # Example representation
    def initialize_population(self):
        route, _ = greedy_tsp_with_time_windows(self.N, self.travel_time, self.time_windows)
        self.population.append(route)

        for _ in range(1, self.population_size):
            instance = random.sample(range(1, N + 1), k=N)
            self.population.append( instance )
            # self.population.append( instance )


    def fitness(self, individual):
        total_distance = 0
        total_penalty = 0
        current_time = 0  # Assume the starting time is 0
        last_node = 0 # Start from the first node

        # Iterate through the individual (route) and calculate the total distance and time window violations
        for i in range(len(individual)):
            current_node = individual[i]
            # print(current_node , end="*")

            # Get the distance between last_node and current_node
            distance = self.travel_time[last_node][current_node]
            total_distance += distance
            
            # Update current time
            current_time += distance
            
            # Check if current time exceeds the time window for the current_node
            start_time, end_time, dur = self.time_windows[current_node]  # Get the time window for the node
            if current_time < start_time:
                # If the arrival time is too early, wait until the start time
                current_time = start_time
            elif current_time > end_time:
                # If the arrival time is too late, apply a penalty
                penalty = (current_time - end_time) * 0.5
                total_penalty += penalty

            # Add the service time for the node (if applicable)
            current_time += dur

            last_node = current_node

        # Return the total fitness, considering distance and penalty
        total_fitness = total_distance + total_penalty
        return total_fitness

    def update_fitness_value(self):
        self.fitness_value = []
        for instance in self.population:
            # print(instance)
            self.fitness_value.append(self.fitness(instance))

    def order_crossover(self, parent1, parent2):
        size = len(parent1)
        
        # Step 1: Take the first half of parent1
        crossover_point = size // 2
        offspring = [-1] * size
        
        # Copy the first half from parent1
        offspring[:crossover_point] = parent1[:crossover_point]
        
        # Step 2: Fill the remaining positions with parent2's genes
        p2_idx = 0  # Start from the beginning of parent2
        for i in range(crossover_point, size):
            while parent2[p2_idx] in offspring:
                p2_idx += 1  # Skip already used genes in offspring
            offspring[i] = parent2[p2_idx]
        
        return offspring


    def swap_mutation(self, solution, mutation_rate):
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(solution)), 2)
            solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
        return solution
    
    
    def shuffle_segment_mutation(self, solution, mutation_rate, max_segment_length = 10):
        """
        Perform mutation by randomly shuffling a segment of a random length within the solution.
        
        :param solution: Current solution (tour)
        :param mutation_rate: Probability of applying mutation
        :param max_segment_length: Maximum possible length of the segment to shuffle
        :return: Mutated solution
        """
        if random.random() < mutation_rate:
            # Choose a random segment length between 2 and max_segment_length
            segment_length = random.randint(2, max_segment_length)
            
            # Choose a random start index for the segment to shuffle
            start_idx = random.randint(0, len(solution) - segment_length)
            
            # Extract the segment and shuffle it
            segment = solution[start_idx:start_idx + segment_length]
            random.shuffle(segment)
            
            # Replace the original segment with the shuffled one
            solution[start_idx:start_idx + segment_length] = segment
        
        return solution
    
    def LocalSearch(self, solution):
        """
        Perform a lighter local search using 2-opt.
        :param solution: Current solution (tour)
        :return: Improved solution
        """
        n = len(solution)

        improved = False
        
        for i in range(0, n - 2):
            for j in range(i + 1, n - 1):
                # Perform the 2-opt swap
                new_solution = self.two_opt_move(solution, i, j)
                is_valid, cost = evaluate(new_solution, self.time_windows, self.travel_time)
                if is_valid and self.fitness(new_solution) < self.fitness(solution):
                    solution = new_solution
                    improved = True
                    break
            if improved:
                break

        return solution

    def two_opt_move(self, solution, i, j):
        """
        Perform a 2-opt move on the solution.
        :param solution: Current solution (tour)
        :param i, j: Indices to swap
        :return: New solution after 2-opt move
        """
        new_solution = solution[:]
        new_solution[i:j+1] = reversed(new_solution[i:j+1])
        return new_solution

    def LocalSearch(self, solution):
        return solution
        

    def tournament_selection(self, tournament_size):
        selected = random.sample(list(zip(self.population, self.fitness_value)), tournament_size)
        return min(selected, key=lambda x: x[1])[0]

    def GA(self):
        # Initialize the population (assuming self.population is already initialized)

        # Create a new population by selecting parents and generating offspring
        new_population = []
        for _ in range(self.population_size // 2):  # Doubling the population size
            # Selection: Tournament Selection
            parent1 = self.tournament_selection(self.tournament_size)
            parent2 = self.tournament_selection(self.tournament_size)

            # Crossover
            offspring1 = self.order_crossover(parent1, parent2)
            offspring2 = self.order_crossover(parent2, parent1)

            # Mutation
            offspring1 = self.swap_mutation(offspring1, self.mutation_rate)
            offspring2 = self.swap_mutation(offspring2, self.mutation_rate)
            offspring1 = self.shuffle_segment_mutation(offspring1, self.mutation_rate)
            offspring2 = self.shuffle_segment_mutation(offspring2, self.mutation_rate)
            _, offspring1 = self.repair_solution(offspring1)
            _, offspring2 = self.repair_solution(offspring2)



            # Apply Local Search to the offspring
            # offspring1 = self.LocalSearch(offspring1)
            # offspring2 = self.LocalSearch(offspring2)

            # Add the offspring to the new population
            new_population.extend([offspring1, offspring2])

        # Combine the current population and the new population
        combined_population = self.population + new_population


        # Apply Local Search to the current population as well (elitism or for optimization)
        for i in range(len(self.population)):
            self.population[i] = self.LocalSearch(self.population[i])

        # Evaluate the fitness of the combined population
        # print("cf")
        # print(combined_population, end="cp")
        fitness_values = [self.fitness(individual) for individual in combined_population]

        # Sort the combined population based on fitness (ascending or descending based on your objective)
        sorted_population = [x for _, x in sorted(zip(fitness_values, combined_population))]

        # Keep the best 'elitism_size' individuals from the sorted population
        self.population = sorted_population[:self.population_size - self.elitism_size]

        # Add the top 'elitism_size' individuals from the last generation
        best_individuals = sorted_population[:self.elitism_size]
        self.population.extend(best_individuals)

        # Track the best solution
        best_fitness = min(self.fitness_value)
        best_solution_idx = self.fitness_value.index(best_fitness)

        # Return the best solution
        return self.population[best_solution_idx]

    def repair_solution(self, route, max_depth = 10):
        """
        Repairs a TSP-TW route to ensure time window feasibility.

        Parameters:
            route (list): Current route as a list of nodes.
            time_windows (dict): A dictionary where keys are nodes, and values are (start, end) time tuples.
            distance_matrix (dict): Distance matrix with travel times between nodes.

        Returns:
            list: Feasible route after repair.
        """
        if (max_depth == 0):
            return False, route
        repaired_route = []
        ready_time = []
        current_time = 0

        for i, node in enumerate(route):
            start_time, end_time, dur = time_windows[node]
            
            # Check if arrival time at this node violates the time window
            if current_time < start_time:
                # Arriving too early: Wait until the window opens
                current_time = start_time
            
            elif current_time > end_time:
                # Arriving too late: Attempt to repair
                # Move the node to a feasible position later in the route
                delayed_node = node
                insert_idx = 0
                for j in range(i):
                    if ready_time[j] + self.travel_time[ route[j] ][delayed_node] <= end_time:
                        insert_idx = j + 1
                route.pop(i)  # Temporarily remove the node
                # insert_idx = random.randint(0, insert_idx)
                route.insert(insert_idx, node)  # Reinsert it at the end
                return self.repair_solution(route, max_depth - 1)  # Recursive repair

            # Add the node to the repaired route and update the current time
            repaired_route.append(node)
            if i < len(route) - 1:
                travel_time = self.travel_time[node][route[i + 1]]
                current_time += travel_time + dur
                ready_time.append(current_time)

        return True, repaired_route
    
    def re_repair(self, solution, iter):
        for _ in range(iter):
            completed, solution = self.repair_solution(solution)
            if completed:
                break
        
        return solution

    def Solve(self):
        for gen in range(self.generations):
            solution = self.GA()
            
            valid_, cost = evaluate(solution, self.time_windows, self.travel_time)

            if valid_:
                best_solution = solution
            
            
            # You may want to track and print the best solution so far
            print(f"Generation {gen + 1}: Best Solution: {best_solution}")
        
            # print(self.fitness(best_solution))
        # Optionally, return the best solution found after all generations
        return best_solution


if __name__ == '__main__':


    N, time_windows, travel_time = read_input(True, "MiniProjectOptimize\TestCase\Subtask_1000\input1.txt")
    


    # Example to initialize and solve the problem
    tsp_solver = GASolves(
        N = N,
        time_windows=time_windows,
        travel_time=travel_time,
        population_size=500, 
        generations=500, 
        mutation_rate=0.5, 
        tournament_size=10, 
        elitism_size=0
    )


    # Solve the TSP problem
    best_solution = tsp_solver.Solve()
    print("Best Solution after all generations:", best_solution)
    print(evaluate(best_solution, time_windows, travel_time))

