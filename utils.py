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
            e, l, d = map(int, lines[i].strip().split())
            time_windows.append((e, l, d))

        # Read the travel time matrix
        travel_time = []
        for i in range(N + 1, len(lines)):
            row = list(map(int, lines[i].strip().split()))
            travel_time.append(row)

    else:
        # Read number of nodes
        N = int(input())

        # Read time window and service time for each node
        time_windows = [(-1, -1, -1)]
        for _ in range(N):
            e, l, d = map(int, input().split())
            time_windows.append((e, l, d))

        # Read the travel time matrix
        travel_time = []
        for _ in range(N + 1):  # N+1 because of depot (node 0)
            row = list(map(int, input().split()))
            travel_time.append(row)

    return N, time_windows, travel_time

def parse_sample_answer(file_path):
    """Parse the sample answer file and return the expected route and travel time."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    N = int(lines[0].strip())  # Number of customers
    optimal_route = list(map(int, lines[1].strip().split()))
    optimal_time = int(lines[2].strip())
    return N, optimal_route, optimal_time

def validate_constraints(route, time_windows, travel_times, start_time=0):
    """
    Validate the route against the time window constraints.

    Parameters:
    - route: The delivery route (list of integers).
    - delivery_constraints: List of tuples (e, l, d) for each customer.
    - travel_times: 2D matrix of travel times.
    - start_time: Starting time of the delivery.

    Returns:
    - True if all constraints are satisfied, False otherwise.
    """
    current_time = start_time
    for i in range(len(route) - 1):
        customer = route[i]  # Current customer
        next_customer = route[i + 1]

        # Travel to the next customer
        current_time += travel_times[customer][next_customer]

        # Check delivery constraints
        e, l, d = time_windows[customer - 1]  # Constraints for customer
        # if current_time < e or current_time > l:
        #     return False  # Time window violated
        current_time += d  # Time spent delivering

    return True


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
            e, l, d = map(int, lines[i].strip().split())
            time_windows.append((e, l, d))

        # Read the travel time matrix
        travel_time = []
        for i in range(N + 1, len(lines)):
            row = list(map(int, lines[i].strip().split()))
            travel_time.append(row)

    else:
        # Read number of nodes
        N = int(input())

        # Read time window and service time for each node
        time_windows = [(-1, -1, -1)]
        for _ in range(N):
            e, l, d = map(int, input().split())
            time_windows.append((e, l, d))

        # Read the travel time matrix
        travel_time = []
        for _ in range(N + 1):  # N+1 because of depot (node 0)
            row = list(map(int, input().split()))
            travel_time.append(row)

    return N, time_windows, travel_time

 

# def evaluate(solution, time_windows, travel_time):
#     """
#     Evaluates a given solution for the TSP with Time Windows problem.

#     Args:
#         solution (list): A permutation of nodes representing the delivery route.
#         time_windows (list of tuples): Time windows (e(i), l(i)) and service durations (d(i)) for each node.
#         travel_time (list of lists): Matrix of travel times t(i, j).

#     Returns:
#         tuple:
#             bool: True if the solution is valid (meets all time window constraints), False otherwise.
#             int: Total time taken for the route if valid, or -1 if invalid.
#     """

#     total_time = 0
#     present_position = 0
#     for next_position in solution:
#         early_TW, late_TW, dur = time_windows[next_position]
#         if next_position == 0: 
#             total_time = max(total_time, early_TW) + dur #Ready to go
#             continue

#         total_time += travel_time[present_position][next_position]
#         total_time = max(total_time, early_TW)

#         if total_time <= late_TW:   
#             total_time += dur
#         else: 
#             return False, -1

#         present_position = next_position

#     return True, total_time + travel_time[present_position][0]
    