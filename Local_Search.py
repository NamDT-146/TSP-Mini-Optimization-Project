import sys
import random
from copy import deepcopy
from utils import evaluate, read_input

def generate_initial_solution(N, time_windows, travel_time):
    """
    Generates an initial feasible solution using a greedy approach.

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
                feasible_customers.append(customer)

        if not feasible_customers:
            # No feasible customer to visit next
            return None

        # Select the customer with the earliest latest time (l)
        next_customer = min(feasible_customers, key=lambda x: time_windows[x][1])
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

    for iteration in range(max_iterations):
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

    return best_solution, best_time


def main():
    # Read input
    N, time_windows, travel_time = read_input(True, "TestCase/Subtask_100/rc_201.1.txt")

    # Generate initial solution
    initial_solution = generate_initial_solution(N, time_windows, travel_time)
    if initial_solution is None:
        print("No feasible initial solution found.")
        return

    # Perform Local Search using S-Improvement
    best_solution, best_time = local_search(N, time_windows, travel_time, initial_solution)

    if best_solution is None:
        print("No feasible solution found during Local Search.")
    else:
        # Output the best solution
        print(N)
        print(' '.join(map(str, best_solution)))
        # Also print the cost (total time) of this best solution
        print("Cost:", best_time)


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
