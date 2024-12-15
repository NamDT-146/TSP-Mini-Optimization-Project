import random
from utils import evaluate, read_input
from Greedy import greedy_tsp_with_time_windows

class GASolves():
        
    def __init__(self, N, time_windows, travel_time, population_size, generations, mutation_rate, tournament_size, elitism_size):
        self.N = N
        self.population_size = population_size
        self.time_windows = time_windows
        self.travel_time = travel_time
        self.max_endtime = max( [l for e, l, d in self.time_windows] )
        self.population = []
        self.initialize_population()
        self.fitness_value = []
        # print(self.population)
        self.update_fitness_value()
        self.penalty_factor = 2
        # print(self.fitness_value)
        self.generations = generations
        self.mutation_rate = mutation_rate 
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.mutation_length = 10

    def add_random_individuals(self, num_random_individuals):
        """ Add random individuals to the population to increase diversity. """
        self.population = self.population[num_random_individuals:]
        for _ in range(num_random_individuals):
            random_solution = random.sample(range(1, self.N + 1), k=self.N)
            self.population.append(random_solution)
    
    # Example representation
    def initialize_population(self):
        # route, _ = greedy_tsp_with_time_windows(self.N, self.travel_time, self.time_windows)
        # self.population.append(route)

        for _ in range( self.population_size):
            instance = random.sample(range(1, N + 1), k=N)
            self.population.append( instance )
            # self.population.append( instance )


    def fitness(self, individual):
        total_distance = 0
        total_penalty = 0
        current_time = 0  # Assume the starting time is 0
        last_node = 0 # Start from the first node
        rm_node = self.N
        save_rm_node = -1
        mn_end_time = self.max_endtime

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
                penalty = (current_time - end_time)
                total_penalty += penalty
                current_time
                save_rm_node = max(save_rm_node, rm_node)
            
            if (save_rm_node != -1):
                    mn_end_time = min(mn_end_time, end_time)

            # Add the service time for the node (if applicable)
            current_time += dur
            rm_node -= 1
            last_node = current_node

        current_time += self.travel_time[current_node][0]

        # Return the total fitness, considering distance and penalty
        total_fitness =  (save_rm_node + 1) * 1000 + (total_penalty) / self.N + 0.25 * (self.max_endtime - mn_end_time) / self.N + current_time / self.N
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


    def swap_segment_mutation(self, solution, mutation_rate, mutation_length):
        """
        Performs a mutation by swapping two random segments of random length in the solution.

        Parameters:
            solution (list): The current solution (route).
            mutation_rate (float): The probability of performing the mutation.

        Returns:
            list: The mutated solution.
        """
        if random.random() < mutation_rate:
            size = len(solution)
            
            # Determine a random length for the segment
            segment_length = random.randint(1, min(mutation_length, self.N // 4))  # Ensure the length is reasonable
            
            # Select two random starting indices
            start1 = random.randint(0, size - segment_length)
            start2 = random.randint(0, size - segment_length)
            
            # Ensure the segments do not overlap
            while abs(start1 - start2) < segment_length:
                start2 = random.randint(0, size - segment_length)
            
            # Swap the segments
            segment1 = solution[start1:start1 + segment_length]
            segment2 = solution[start2:start2 + segment_length]
            
            solution[start1:start1 + segment_length] = segment2
            solution[start2:start2 + segment_length] = segment1

        return solution

    
    
    def shuffle_segment_mutation(self, solution, mutation_rate, max_segment_length = 20):
        """
        Perform mutation by randomly shuffling a segment of a random length within the solution.
        
        :param solution: Current solution (tour)
        :param mutation_rate: Probability of applying mutation
        :param max_segment_length: Maximum possible length of the segment to shuffle
        :return: Mutated solution
        """
        if random.random() < mutation_rate:
            max_segment_length = min(max_segment_length, self.N // 4)
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
    
    # def LocalSearch(self, solution):
    #     """
    #     Perform a lighter local search using 2-opt.
    #     :param solution: Current solution (tour)
    #     :return: Improved solution
    #     """
    #     n = len(solution)

    #     improved = False
        
    #     for i in range(0, n - 2):
    #         for j in range(i + 1, n - 1):
    #             # Perform the 2-opt swap
    #             new_solution = self.two_opt_move(solution, i, j)
    #             is_valid, cost = evaluate(new_solution, self.time_windows, self.travel_time)
    #             if is_valid and self.fitness(new_solution) < self.fitness(solution):
    #                 solution = new_solution
    #                 improved = True
    #                 break
    #         if improved:
    #             break

    #     return solution

    # def two_opt_move(self, solution, i, j):
    #     """
    #     Perform a 2-opt move on the solution.
    #     :param solution: Current solution (tour)
    #     :param i, j: Indices to swap
    #     :return: New solution after 2-opt move
    #     """
    #     new_solution = solution[:]
    #     new_solution[i:j+1] = reversed(new_solution[i:j+1])
    #     return new_solution

    def LocalSearch(self, solution):
        return self.relocate_and_remove_infeasible_nodes(solution)
        # return solution
        

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
            offspring1 = self.swap_segment_mutation(offspring1, self.mutation_rate, self.mutation_length)
            offspring2 = self.swap_segment_mutation(offspring2, self.mutation_rate, self.mutation_length)
            offspring1 = self.shuffle_segment_mutation(offspring1, self.mutation_rate, self.mutation_length)
            offspring2 = self.shuffle_segment_mutation(offspring2, self.mutation_rate, self.mutation_length)
            
            # Apply Local Search to the offspring
            offspring1 = self.LocalSearch(offspring1)
            offspring2 = self.LocalSearch(offspring2)
            
            offspring1 = self.improve(offspring1, 1)
            offspring2 = self.improve(offspring2, 1)



            

            # Add the offspring to the new population
            new_population.extend([offspring1, offspring2])

        # Combine the current population and the new population
        combined_population = self.population + new_population


        # Apply Local Search to the current population as well (elitism or for optimization)
        # for i in range(len(self.population)):
        #     self.population[i] = self.LocalSearch(self.population[i])

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

    def repair_solution(self, route, best_fitness, max_depth = 20):
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
        
        if (self.fitness(route) < best_fitness):
            return True, route

        repaired_route = []
        ready_time = []
        current_time = 0

        for i in range(len(route)):
            node = route[i]
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
                
                return self.repair_solution(route, best_fitness, max_depth - 1)  # Recursive repair

            # Add the node to the repaired route and update the current time
            repaired_route.append(node)
            if i < len(route) - 1:
                travel_time = self.travel_time[node][route[i + 1]]
                current_time += travel_time + dur
                ready_time.append(current_time)

        return True, route
    
    def improve(self, solution, iter):
        initial_solution = solution
        best_fitness = self.fitness(initial_solution)
        for _ in range(iter):
            completed, solution = self.repair_solution(solution, best_fitness)
            if completed:
                return solution
                
        
        return initial_solution

    def Solve(self):
        for gen in range(self.generations):
            solution = self.GA()
            best_solution = None
            
            valid_, cost = evaluate(solution, self.time_windows, self.travel_time)

            if valid_:
                best_solution = solution
            
            
            # You may want to track and print the best solution so far
            print(f"Generation {gen + 1}: Best Solution: {best_solution}")
        
            print(self.fitness(solution))
            if best_solution is None:
                best_solution = solution
        # Optionally, return the best solution found after all generations
            if (gen + 1) % 30 == 0:
                self.add_random_individuals(30)
        return best_solution

    def relocate_and_remove_infeasible_nodes(self, route):
        """
        Relocates nodes to feasible positions or removes them if no feasible position exists.

        Parameters:
            route (list): Current route as a list of nodes.

        Returns:
            list: Feasible route after repairing (infeasible nodes are excluded).
        """
        n = len(route)
        current_time = 0
        current_node = 0
        gap_sum = 0
        gap_count = 0
        infeasible_node = []
        feasible_route = []

        # Precompute end times for the current route
        for i in range(n):
            next_node = route[i]
            start_time, end_time, duration = self.time_windows[next_node]
            current_time += self.travel_time[current_node][next_node]
            if current_time < start_time:
                gap_sum += start_time - current_time
                current_time = start_time
            if current_time > end_time:
                infeasible_node.append(next_node)
                if gap_count != 0:
                    current_time += gap_sum / gap_count
            else:
                gap_sum += self.travel_time[current_node][next_node] + duration
                gap_count += 1
                feasible_route.append(next_node)
                current_time += duration

        repaired_route = self.repair_with_two_pointers(feasible_route, infeasible_node)

        return repaired_route


    def repair_with_two_pointers(self, feasible_route, infeasible_nodes):
        """
        Repairs a TSP-TW route by merging a feasible route and sorted infeasible nodes
        using a two-pointer approach, appending infeasible nodes when necessary.

        Parameters:
            feasible_route (list): List of nodes that already form a feasible route.
            infeasible_nodes (list): List of nodes that are infeasible in the initial solution.

        Returns:
            list: A new route combining feasible nodes and repaired infeasible nodes.
        """
        # Sort infeasible nodes by their end time
        infeasible_nodes.sort(key=lambda node: self.time_windows[node][1])  # Sort by end_time

        # Initialize pointers
        feasible_idx = 0
        infeasible_idx = 0
        current_time = 0  # Track current time
        new_route = []  # Final repaired route
        end_times = []  # Track end times in new route

        while feasible_idx < len(feasible_route) or infeasible_idx < len(infeasible_nodes):
            if feasible_idx < len(feasible_route):
                feasible_node = feasible_route[feasible_idx]
                feasible_start, feasible_end, feasible_duration = self.time_windows[feasible_node]

                # Calculate arrival time at feasible node
                if len(new_route) == 0:
                    feasible_arrival = self.travel_time[0][feasible_node]
                else:
                    prev_node = new_route[-1]
                    feasible_arrival = end_times[-1] + self.travel_time[prev_node][feasible_node]

                # Process infeasible node
                if infeasible_idx < len(infeasible_nodes):
                    infeasible_node = infeasible_nodes[infeasible_idx]
                    infeasible_start, infeasible_end, infeasible_duration = self.time_windows[infeasible_node]

                    # Calculate arrival time at infeasible node
                    if len(new_route) == 0:
                        infeasible_arrival = self.travel_time[0][infeasible_node]
                    else:
                        prev_node = new_route[-1]
                        infeasible_arrival = end_times[-1] + self.travel_time[prev_node][infeasible_node]

                    # Check infeasible node feasibility
                    if infeasible_arrival <= infeasible_end:
                        # Append infeasible node if it cannot be placed after the current feasible node
                        if infeasible_arrival + infeasible_duration > feasible_arrival:
                            new_route.append(infeasible_node)
                            current_time = max(infeasible_start, infeasible_arrival) + infeasible_duration
                            end_times.append(current_time)
                            infeasible_idx += 1
                            continue  # Skip feasible node in this iteration

                # Otherwise, append feasible node
                new_route.append(feasible_node)
                current_time = max(feasible_start, feasible_arrival) + feasible_duration
                end_times.append(current_time)
                feasible_idx += 1

            elif infeasible_idx < len(infeasible_nodes):
                # Append remaining infeasible nodes
                infeasible_node = infeasible_nodes[infeasible_idx]
                infeasible_start, infeasible_end, infeasible_duration = self.time_windows[infeasible_node]

                if len(new_route) == 0:
                    infeasible_arrival = self.travel_time[0][infeasible_node]
                else:
                    prev_node = new_route[-1]
                    infeasible_arrival = end_times[-1] + self.travel_time[prev_node][infeasible_node]

                # Append infeasible node
                new_route.append(infeasible_node)
                current_time = max(infeasible_start, infeasible_arrival) + infeasible_duration
                end_times.append(current_time)
                infeasible_idx += 1

        return new_route



    # def repair_solution(self, route, max_depth=120):
    #     """
    #     Repairs a TSP-TW route iteratively to ensure time window feasibility.

    #     Parameters:
    #         route (list): Current route as a list of nodes.
    #         max_depth (int): Maximum attempts to repair the route.

    #     Returns:
    #         tuple: (bool, list) - Feasibility status and repaired route.
    #     """
    #     repaired_route = []
    #     ready_time = []
    #     current_time = 0
    #     depth_attempts = 0
    #     i = 0  # Start from the first node

    #     while i < len(route):
    #         if depth_attempts >= max_depth:
    #             return False, route  # Abort repair if max depth is reached

    #         node = route[i]
    #         start_time, end_time, duration = self.time_windows[node]

    #         # Check if arrival time violates the time window
    #         if current_time < start_time:
    #             # Arrive too early: Wait until the window opens
    #             current_time = start_time

    #         elif current_time > end_time:
    #             # Arrive too late: Attempt to repair
    #             delayed_node = node
    #             insert_idx = 0

    #             # Find a valid position to reinsert the node
    #             for j in range(i):
    #                 if (
    #                     ready_time[j] + self.travel_time[route[j]][delayed_node]
    #                     <= end_time
    #                 ):
    #                     insert_idx = j + 1

    #             # If no valid position, return infeasibility
    #             if insert_idx == -1:
    #                 return False, route

    #             # Remove the delayed node and reinsert it at a feasible position
    #             route.pop(i)
    #             route.insert(insert_idx, delayed_node)
    #             depth_attempts += 1  # Increment repair attempts

    #             # Start repair from the next node after insertion
    #             i = insert_idx + 1
    #             current_time = ready_time[insert_idx - 1] + self.travel_time[route[insert_idx - 1]][delayed_node] + duration
    #             continue  # Skip the increment for i

    #         # Node is feasible; add to repaired route
    #         repaired_route.append(node)
    #         if i < len(route) - 1:
    #             travel_time = self.travel_time[node][route[i + 1]]
    #             current_time += travel_time + duration
    #             ready_time.append(current_time)

    #         i += 1  # Proceed to the next node

    #     return True, repaired_route


if __name__ == '__main__':


    N, time_windows, travel_time = read_input(True, "TestCase\Subtask_100\\N20ft304.dat")
    


    # Example to initialize and solve the problem
    tsp_solver = GASolves(
        N = N,
        time_windows=time_windows,
        travel_time=travel_time,
        population_size=500, 
        generations=1000, 
        mutation_rate=0.5, 
        tournament_size=8, 
        elitism_size=2
    )


    # Solve the TSP problem
    best_solution = tsp_solver.Solve()
    print("Best Solution after all generations:", best_solution)
    print(evaluate(best_solution, time_windows, travel_time))
    # route = [1, 18, 92, 88, 48, 69, 24, 73, 9, 33, 39, 61, 60, 32, 19, 95, 23, 28, 79, 89, 25, 38, 7, 47, 26, 15, 99, 55, 76, 5, 41, 37, 6, 72, 36, 31, 30, 3, 64, 42, 80, 67, 35, 46, 57, 75, 49, 2, 16, 14, 97, 85, 29, 56, 98, 100, 87, 52, 51, 11, 71, 65, 21, 44, 74, 70, 53, 20, 4, 54, 82, 50, 84, 17, 78, 8, 10, 96, 83, 27, 59, 13, 45, 40, 22, 90, 58, 63, 93, 94, 68, 43, 34, 66, 91, 62, 86, 12, 81, 77]
    # print(tsp_solver.fitness(route))
    # _, solution = tsp_solver.repair_solution(route, tsp_solver.fitness(route), 100)    
    # print(solution, tsp_solver.fitness(solution), _)
    # 83235.0
