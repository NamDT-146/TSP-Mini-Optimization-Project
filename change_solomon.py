import math
import numpy as np


def read_solomon_dataset(file_path, N):
    """
    Convert a Solomon dataset to the TSP with time windows format.

    Parameters:
    - file_path: Path to the Solomon dataset file.
    - N: Number of customers to include in the test case.

    Returns:
    - A string formatted like your test cases.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse depot and customer data
    locations = []
    time_windows = []
    service_times = []
    
    for line in lines[1:N + 2]:  # Skip the header, process depot + first N customers
        parts = line.split()
        x, y = float(parts[1]), float(parts[2])
        e, l = int(parts[4]), int(parts[5])  # Time window
        d = int(parts[6])  # Service time
        locations.append((x, y))
        time_windows.append((e, l))
        service_times.append(d)
    
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
        e, l = time_windows[i]
        d = service_times[i]
        test_case.append(f"{e} {l} {d}")
    for row in travel_time_matrix:
        test_case.append(" ".join(map(str, row)))
    
    return "\n".join(test_case)


# Example Usage
file_path = "E:\\test_case\\rc2\\rc208.txt"  # Replace with the actual Solomon dataset file path
test_case = read_solomon_dataset(file_path, 100)  # Generate a test case for N=100

# Save the test case to a file
with open("test_case_solomon_208.txt", "w") as f:
    f.write(test_case)
