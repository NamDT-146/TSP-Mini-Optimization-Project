import random
from utils import evaluate, read_input
from Greedy import greedy_tsp_with_time_windows
import time


class GASolves():
        
    def __init__(self, N, time_windows, travel_time, population_size, generations, mutation_rate, tournament_size, elitism_size, time_out):
        self.N = N
        self.population_size = population_size
        self.time_windows = time_windows
        self.travel_time = travel_time
        self.max_endtime = max( [l for e, l, d in self.time_windows] )
        # Flatten the list of lists into a single list
        all_elements = [item for inner_list in self.travel_time for item in inner_list]
        # Overall average
        self.mean_distance = sum(all_elements) / len(all_elements)
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
        self.mutation_length = 5
        self.start_time = time.process_time()
        # set the proper time to run algorithm
        self.end_time = self.start_time + time_out
         
        

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

        feasible_loss =  (save_rm_node + 1) * self.mean_distance * 2 +  (total_penalty) / self.N + 0.25 * (self.max_endtime - mn_end_time) / self.N  
        # Return the total fitness, considering distance and penalty
        total_fitness =  feasible_loss * 5 + current_time / self.N
        # total_fitness =  (total_penalty) / self.N + 0.25 * (self.max_endtime - mn_end_time) / self.N + current_time / self.N
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
            segment_length = 2
            if max_segment_length >= 2:
                segment_length = random.randint(2, max_segment_length)
            
            # Choose a random start index for the segment to shuffle
            start_idx = random.randint(0, len(solution) - segment_length)
            
            # Extract the segment and shuffle it
            segment = solution[start_idx:start_idx + segment_length]
            random.shuffle(segment)
            
            # Replace the original segment with the shuffled one
            solution[start_idx:start_idx + segment_length] = segment
        
        return solution

    def Growth(self, solution):
        """
        Apply local search to improve the fitness of a solution in TSP-TW.

        Args:
            solution (list): Current route as a list of nodes.
            time_windows (dict): Dictionary of nodes with (earliest, latest) time windows.
            travel_time (dict): Travel time matrix.
            fitness_function (function): Function to evaluate the fitness of a solution.

        Returns:
            list: Improved solution if fitness improves, otherwise the original.
        """
        current_fitness = self.fitness(solution)
        improved_solution = self.LocalSearch(solution)

        # Calculate fitness of the improved solution
        improved_fitness = self.fitness(improved_solution)

        # Return the improved solution only if fitness improves
        if improved_fitness < current_fitness:  # Minimize fitness
            return improved_solution
        return solution

    # def LocalSearch(self, route):
    #     """
    #     Applies fitness-guided 2-Opt local search for TSP-TW.

    #     Args:
    #         route (list): Current route as a list of nodes.
    #         time_windows (dict): Dictionary of nodes with (earliest, latest) time windows.
    #         travel_time (dict): Travel time matrix.
    #         fitness_function (function): Function to evaluate the fitness of a solution.

    #     Returns:
    #         list: Improved route if fitness improves, otherwise the original.
    #     """
    #     current_fitness = self.fitness(route)
    #     best_route = route[:]
        
    #     for i in range(1, len(route) - 2):  # Avoid depot at start and end
    #         for j in range(i + 1, len(route) - 1):
    #             # Perform a 2-opt swap
    #             new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]

                
    #             new_fitness = self.fitness(new_route)

    #             # Update if fitness improves
    #             if new_fitness < current_fitness:
    #                 current_fitness = new_fitness
    #                 best_route = new_route
    #     return best_route


    def LocalSearch(self, route, num_trials=3):
        """
        Randomized O(1) local search: Relocate a single random node to a better position.

        Args:
            route (list): Current route as a list of nodes.
            time_windows (dict): Dictionary of nodes with (earliest, latest) time windows.
            travel_time (dict): Travel time matrix.
            fitness_function (function): Function to evaluate the fitness of a solution.
            num_trials (int): Number of random relocation trials to perform.

        Returns:
            list: Improved route if fitness improves, otherwise the original route.
        """
        current_fitness = self.fitness(route)
        n = len(route)
        
        for _ in range(num_trials):  # Perform a limited number of random trials
            i = random.randint(n // 2 - 1, n - 1)  # Random node to relocate (skip depot)
            node = route.pop(i)  # Remove the node temporarily
            
            # Try inserting at random positions
            best_position = i
            best_fitness = current_fitness
            for _ in range(num_trials):  # Try a small random subset of positions
                j = random.randint(0, n - 1)  # Random position (skip depot positions)
                route.insert(j, node)
                
                new_fitness = self.fitness(route)
                if new_fitness < best_fitness:  # Check if fitness improves
                    best_position = j
                    best_fitness = new_fitness
                
                route.pop(j)  # Undo insertion
            
            # Insert the node back at the best position found
            route.insert(best_position, node)
            current_fitness = best_fitness
        
        return route


    # def LocalSearch(self, solution):
        
    #     return solution
        

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

            offspring1 = self.improve(offspring1, 5, 1)
            offspring2 = self.improve(offspring2, 5, 1)
            # Apply Local Search to the offspring

            offspring1 = self.Growth(offspring1)
            offspring2 = self.Growth(offspring2)
            


            

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

    def repair_solution(self, route, best_fitness, max_depth = 10):
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
    
    def improve(self, solution, max_depth, iter):
        best_solution = solution
        best_fitness = self.fitness(best_solution)
        completed, solution = self.repair_solution(solution, best_fitness, max_depth)
        if completed:
            best_solution = solution
            best_fitness = self.fitness(best_solution)
            
        for _ in range(iter):
            solution = self.relocate_and_remove_infeasible_nodes(solution)
            if self.fitness(solution) < best_fitness:
                best_solution = solution
                best_fitness = self.fitness(best_solution)
                
        
        return best_solution

    def Solve(self):
        Feasible = False
        best_solution = None
        stable_step = 0
        local_best_solution = None
        reset = True
        for gen in range(self.generations):
            solution = self.GA()
            if gen == 0:
                best_solution = solution
            if reset:
                local_best_solution = solution
                reset = False
            
            valid_, cost = evaluate(solution, self.time_windows, self.travel_time)
            if valid_:
                Feasible = True

            self.fitness(solution)
            # You may want to track and print the best solution so far
            print(f"Generation {gen + 1}: Best Solution: {best_solution} Feasible: {Feasible}")
        
            print(self.fitness(local_best_solution))
            if self.fitness(local_best_solution) > self.fitness(solution):
                local_best_solution = solution
                stable_step = 0
            else:
                stable_step += 1

            if self.fitness(best_solution) > self.fitness(solution):
                best_solution = solution

            
            
        # Optionally, return the best solution found after all generations
            if (stable_step + 1) % 10 == 0:
                self.add_random_individuals(50)
            if (stable_step)  == 20:
                self.mutation_rate = 0.5
            if (stable_step) == 50:
                self.mutation_rate = 1
            if stable_step == 100:
                self.mutation_rate = 0.2
                stable_step = 0
                reset = True
                self.add_random_individuals(self.population_size)
            if time.process_time() > self.end_time:
                break
            
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


if __name__ == '__main__':


    N, time_windows, travel_time = read_input(True, "TestCase\Subtask_100\\n80w60.001.txt")
    


    # Example to initialize and solve the problem
    tsp_solver = GASolves(
        N = N,
        time_windows=time_windows,
        travel_time=travel_time,
        population_size=1000, 
        generations=999999, 
        mutation_rate=0.2, 
        tournament_size=8, 
        elitism_size=2, 
        time_out=180 #second
    )


    # Solve the TSP problem
    best_solution = tsp_solver.Solve()
    print("Best Solution after all generations:", best_solution)
    print(evaluate(best_solution, time_windows, travel_time))
    # route = [2, 3, 29, 1, 5, 6, 7, 12, 8, 16, 9, 17, 10, 18, 11, 19, 20, 13, 21, 14, 22, 15, 24, 25, 27, 28, 30, 23, 31, 33, 26, 34, 35, 36, 32, 39, 41, 42, 43, 44, 37, 38, 40, 47, 48, 49, 50, 45, 46, 4]
    # print(tsp_solver.fitness(route))
    # solution = tsp_solver.relocate_and_remove_infeasible_nodes(route)  

    # print(solution, tsp_solver.fitness(solution), tsp_solver.fitness(route))
    # 83235.0

    
