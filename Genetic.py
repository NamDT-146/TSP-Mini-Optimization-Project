from utils import read_input

def greedy_tsp_with_time_windows(N, travel_time, time_windows):
    """
    Solve the TSPTW problem using a greedy approach.

    Args:
        num_cities (int): Number of cities (including the depot).
        distance_matrix (list of lists): Matrix of travel times between cities.
        time_windows (list of tuples): Time window constraints for each city (start_time, end_time).

    Returns:
        list: The tour as a sequence of city indices.
        float: The total travel time or a penalty value if infeasible.
    """
    # Initialization
    current_city = 0
    current_time = 0
    visited = set([current_city])
    tour = [current_city]
    total_time = 0

    while len(visited) < N:
        # Find the next best city
        best_next_city = None
        best_travel_time = float('inf')

        for next_city in range(1, N + 1):
            if next_city not in visited:
                # travel_time = travel_time[current_city][next_city]
                arrival_time = current_time + travel_time[current_city][next_city]
                start_time, end_time, dur = time_windows[next_city]

                # Check time window feasibility
                if arrival_time <= end_time:  # Feasible to arrive within the time window
                    adjusted_time = max(arrival_time, start_time) + dur
                    if travel_time < best_travel_time:
                        best_next_city = next_city
                        best_travel_time = travel_time

        # If no feasible city found, return failure
        if best_next_city is None:
            print("No feasible solution found.")
            return tour, float('inf')

        # Visit the best city
        tour.append(best_next_city)
        visited.add(best_next_city)
        current_time = max(current_time + best_travel_time, time_windows[best_next_city][0])
        total_time += best_travel_time
        current_city = best_next_city

    # Return to depot
    total_time += travel_time[current_city][0]
    tour.append(0)

    return tour, total_time


# Example usage
if __name__ == "__main__":
    N, travel_time, time_windows = read_input(True, "MiniProjectOptimize\TestCase\Subtask_10\input2.txt")

    tour, total_time = greedy_tsp_with_time_windows(N, travel_time, time_windows)
    print("Tour:", tour)
    print("Total Time:", total_time)
