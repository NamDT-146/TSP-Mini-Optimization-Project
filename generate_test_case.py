import math
import random
import numpy as np

def generate_large_scale_tsp_tw(N=1000, max_time_window=10000, max_travel_time=500):
    """
    Generate a synthetic dataset for TSP with Time Windows (TSP-TW).

    Parameters:
    - N: Number of customers (default: 1000).
    - max_time_window: Maximum value for the time window constraints (default: 10000).
    - max_travel_time: Maximum value for travel times (default: 500).

    Returns:
    - A string formatted as a test case.
    """
    # Randomly generate coordinates for depot and customers
    locations = [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(N + 1)]
    
    # Generate random time windows and service times
    delivery_constraints = []
    current_time = 0
    for i in range(N):
        d = random.randint(5, 30)  # Service time
        e = current_time
        l = current_time + max_time_window // 2  # Generous time window
        delivery_constraints.append((e, l, d))
        current_time += d + random.randint(10, max_travel_time // 4)
    
    # Calculate symmetric travel time matrix (Euclidean distances)
    def euclidean_distance(loc1, loc2):
        return int(math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2))
    
    travel_time_matrix = np.zeros((N + 1, N + 1), dtype=int)
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                travel_time_matrix[i][j] = euclidean_distance(locations[i], locations[j])
    
    # Format the test case
    test_case = [str(N)]  # First line: number of customers
    for i in range(1, N + 1):
        e, l, d = delivery_constraints[i - 1]
        test_case.append(f"{e} {l} {d}")
    for row in travel_time_matrix:
        test_case.append(" ".join(map(str, row)))
    
    return "\n".join(test_case)

# Generate a test case for N=1000
test_case_1000 = generate_large_scale_tsp_tw(1000)

# Save the test case to a file
file_path = "solvable_test_case_10000.txt"
with open(file_path, "w") as f:
    f.write(test_case_1000)

file_path
