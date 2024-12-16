import random

def read_input(from_file=False, file_path=None):
    """
    Reads input for the TSP with Time Windows problem.
    
    Args:
        from_file (bool): Whether to read input from a file.
        file_path (str): Path to the input file (if from_file is True).

    Returns:
        N (int): Number of nodes (customers + depot).
        time_windows (list of tuples): List of (e(i), l(i), d(i)) for each node.
        travel_time (list of lists): Matrix of travel times t(i, j).
    """
    if from_file and file_path:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Read number of nodes
        N = int(lines[0].strip())

        # Read time window and service time for each node
        time_windows = [(-1, -1, -1)]
        for i in range(1, N + 1):
            e, l, d = map(float, lines[i].strip().split())
            time_windows.append((e, l, d))

        # Read the travel time matrix
        travel_time = []
        for i in range(N + 1, len(lines)):
            row = list(map(float, lines[i].strip().split()))
            travel_time.append(row)

    else:
        # Read number of nodes
        N = int(input())

        # Read time window and service time for each node
        time_windows = [(-1, -1, -1)]
        for _ in range(N):
            e, l, d = map(float, input().split())
            time_windows.append((e, l, d))

        # Read the travel time matrix
        travel_time = []
        for _ in range(N + 1):  # N+1 because of depot (node 0)
            row = list(map(float, input().split()))
            travel_time.append(row)

    return N, time_windows, travel_time

 

def evaluate(solution, time_windows, travel_time):
    """
    Evaluates a given solution for the TSP with Time Windows problem.

    Args:
        solution (list): A permutation of nodes representing the delivery route.
        time_windows (list of tuples): Time windows (e(i), l(i)) and service durations (d(i)) for each node.
        travel_time (list of lists): Matrix of travel times t(i, j).

    Returns:
        tuple:
            bool: True if the solution is valid (meets all time window constraints), False otherwise.
            int: Total time taken for the route if valid, or -1 if invalid.
    """

    total_time = 0
    present_position = 0
    how_far = 0
    for next_position in solution:
        how_far += 1
        early_TW, late_TW, dur = time_windows[next_position]
        if next_position == 0: 
            total_time = max(total_time, early_TW) + dur #Ready to go
            continue

        total_time += travel_time[present_position][next_position]
        total_time = max(total_time, early_TW)

        if total_time <= late_TW:   
            total_time += dur
        else: 
            return False, -1

        present_position = next_position

    return True, total_time + travel_time[present_position][0]


def generate_initial_solution(N, time_windows, travel_time):
    """
    Generates an initial feasible solution using a greedy approach that considers
    time windows, nearest node information, and a dynamically scaled randomness probability.

    Args:
        N (int): Number of customers.
        time_windows (list of tuples): Time windows and service durations.
        travel_time (list of lists): Travel time matrix.

    Returns:
        list: Initial feasible solution as a permutation of customers.
    """
    unvisited = set(range(1, N + 1))
    current = 0  # Start from depot
    solution = []
    current_time = 0

    while unvisited:
        feasible_customers = []
        for customer in unvisited:
            e, l, d = time_windows[customer]
            arrival_time = current_time + travel_time[current][customer]
            if arrival_time <= l:
                # Calculate the composite score
                distance = travel_time[current][customer]
                time_window_width = l - e
                score = 1 / (distance + 1) + N / (l + 1)
                feasible_customers.append((customer, score))

        if not feasible_customers:
            # No feasible customer to visit next
            print(f"No feasible customers found from current node {current} at time {current_time}.")
            return None

        # Dynamically scale randomness probability based on scores
        scores = [score for _, score in feasible_customers]
        max_score = max(scores)
        min_score = min(scores)
        
        if max_score == min_score:
            # If all scores are the same, assign equal probabilities
            probabilities = [1 / len(feasible_customers)] * len(feasible_customers)
        else:
            total_scaled_prob = sum(scores)
            probabilities = [prob / total_scaled_prob for prob in scores]
        
        print(probabilities)

        # Select the next customer based on the scaled probabilities
        next_customer = random.choices([customer for customer, _ in feasible_customers], probabilities)[0]

        solution.append(next_customer)
        unvisited.remove(next_customer)
        e, l, d = time_windows[next_customer]
        arrival_time = current_time + travel_time[current][next_customer]
        current_time = max(arrival_time, e) + d
        current = next_customer

    return solution


def two_opt_swap(route, i, k):
    """
    Performs a 2-opt swap by reversing the segment between indices i and k.

    Args:
        route (list): Current route.
        i (int): Start index.
        k (int): End index.

    Returns:
        list: New route after the swap.
    """
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route


def S_Improvement(current_solution, time_windows, travel_time):
    """
    Implements S-Improvement selection function:
    - Evaluate all neighbors of the current solution.
    - Find the neighbor with the best improvement.
    - If there's a better neighbor, return it. Otherwise, return None.

    Args:
        current_solution (list): Current route.
        time_windows (list of tuples): Time windows.
        travel_time (list of lists): Travel times.

    Returns:
        list or None: The best improving neighbor or None if no improvement is found.
    """
    is_feasible, current_time = evaluate(current_solution, time_windows, travel_time)
    if not is_feasible:
        # Current solution not feasible, no reason to improve from here.
        return None

    best_neighbor = None
    best_time = current_time
    n = len(current_solution)

    for i in range(n - 1):
        for k in range(i + 1, n):
            new_solution = two_opt_swap(current_solution, i, k)
            feasible, new_time = evaluate(new_solution, time_windows, travel_time)
            if feasible and new_time < best_time:
                best_neighbor = new_solution
                best_time = new_time

    return best_neighbor


def local_search(N, time_windows, travel_time, initial_solution, max_iterations=1000):
    """
    Performs Local Search using the 2-opt method and S-Improvement neighbor selection.

    Args:
        N (int): Number of customers.
        time_windows (list of tuples): Time windows and service durations.
        travel_time (list of lists): Travel time matrix.
        initial_solution (list): Initial feasible solution.
        max_iterations (int): Maximum number of iterations.

    Returns:
        list: Best found solution.
        int: Total time of the best solution.
    """
    best_solution = initial_solution
    is_feasible, best_time = evaluate(best_solution, time_windows, travel_time)
    if not is_feasible:
        print("Initial solution is not feasible.")
        return None, -1
    
    for _ in range(10):
        # Use S-Improvement to select the next solution
        improved_solution = S_Improvement(best_solution, time_windows, travel_time)
        if improved_solution is None:
            # No improving neighbor found
            break
        # Update best_solution and best_time
        is_feasible, new_time = evaluate(improved_solution, time_windows, travel_time)
        if is_feasible and new_time < best_time:
            best_solution = improved_solution
            best_time = new_time
        else:
            # Even if improved_solution is feasible but not better, S_Improvement would not return it.
            # Hence this check is somewhat redundant.
            break

        max_iterations -= 1

    return best_solution, best_time, max_iterations

def Solve_LocalSearch(N, time_windows, travel_time, max_retries=400000):
    # Retry mechanism for generating initial solution
    initial_solution = None
    best_cost = float('inf')
    best_solution = None
    while max_retries > 0:
        initial_solution = generate_initial_solution(N, time_windows, travel_time)
        if initial_solution is not None:
            break

        if initial_solution is None:
            print("No feasible initial solution found after maximum retries.")
            return

        print("Initial solution found:", initial_solution)

        # Perform Local Search using S-Improvement
        local_best_solution, local_best_cost, max_retries = local_search(N, time_windows, travel_time, initial_solution, max_retries)

        if local_best_solution is None:
            print("No feasible solution found during Local Search.")
        else:
            # Output the best solution
            if best_cost > local_best_cost:
                best_cost = local_best_cost
                best_solution = local_best_solution

        max_retries -= 1
    return best_solution


def main():
    # Read input
    N, time_windows, travel_time = read_input(True, 'TestCase/Subtask_10/rc_207.4.txt')

    # Retry mechanism for generating initial solution
    max_retries = 1
    # Perform Local Search using S-Improvement
    best_solution = Solve_LocalSearch(N, time_windows, travel_time, max_retries)

    if best_solution is None:
        print("No feasible solution found during Local Search.")
    else:
        # Output the best solution
        print(N)
        print(' '.join(map(str, best_solution)))


if __name__ == "__main__":
    main()
