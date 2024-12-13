import math
from utils import read_input
from Greedy import greedy_tsp_with_time_windows

def tsp_with_time_windows_backtrack(N, travel_time, time_windows, initial_bound):
    """
    Solve the TSPTW problem using a branch and bound approach with backtracking.

    Args:
        num_cities (int): Number of cities (including the depot).
        distance_matrix (list of lists): Matrix of travel times between cities.
        time_windows (list of tuples): Time window constraints for each city (start_time, end_time).

    Returns:
        list: The optimal tour as a sequence of city indices.
        float: The total travel time of the optimal tour.
    """
    best_tour = None
    best_cost = initial_bound

    def calculate_lower_bound(visited, current_cost):
        """Estimate a lower bound for the remaining path."""
        # Use a simple heuristic: minimum edge from unvisited cities
        bound = current_cost
        for i in range(1, N + 1):
            if i not in visited:
                min_edge = min(travel_time[i][j] for j in range(1, N + 1) if j != i)
                bound += min_edge
        return bound

    def dfs(current_city, visited, current_time, tour):
        nonlocal best_tour, best_cost

        # If all cities are visited, check returning to the depot
        if len(visited) == N:
            final_cost = current_time + travel_time[current_city][0]            
            # print(final_cost)

            if final_cost < best_cost:
                best_cost = final_cost
                best_tour = tour
            return

        # Explore all possible next cities
        for next_city in range(1, N + 1):
            if next_city not in visited:
                start_time, end_time, dur = time_windows[next_city]
                arrival_time = max(current_time + travel_time[current_city][next_city], start_time)

                # Check feasibility
                if arrival_time > end_time:
                    continue

                # Update state and calculate bounds
                next_time = arrival_time + dur
                lower_bound = calculate_lower_bound(visited | {next_city}, next_time)

                # Branch and Bound: prune paths with a bound worse than the current best
                if lower_bound < best_cost:
                    dfs(next_city, visited | set({next_city}), next_time, tour + [next_city])

    # Start DFS from the depot (city 0)
    dfs(0, set({}), 0, [])
    return best_tour, best_cost

def branch_n_bound_TSP(N, travel_time, time_windows):
    greedy_tour, init_upper_bound = greedy_tsp_with_time_windows(N, travel_time, time_windows)
    return tsp_with_time_windows_backtrack(N, travel_time, time_windows, init_upper_bound + 1)




# Example usage
if __name__ == "__main__":
    N, time_windows, travel_time = read_input(True, "MiniProjectOptimize\TestCase\Subtask_10\input2.txt")

    optimal_tour, optimal_cost = branch_n_bound_TSP(N, travel_time, time_windows )
    
    print("Optimal Tour:", optimal_tour)
    print("Optimal Cost:", optimal_cost)
