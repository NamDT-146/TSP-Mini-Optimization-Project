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

        how_far += 1
        present_position = next_position

    return True, total_time + travel_time[present_position][0]
<<<<<<< HEAD
    

if __name__ == "__main__":
    N, time_windows, travel_time = read_input(True, "TestCase\Subtask_100\\rc201.0")

    solution = [int(i) for i in "14 40 78 8 41 71 9 18 10 45 16 64 79 33 62 29 65 24 31 42 36 56 34 48 51 38 74 21 7 67 63 30 76 19 59 5 55 61 17 53 47 44 57 70 73 4 1 69 49 39 52 32 28 58 26 25 72 66 60 54 22 3 75 43 12 20 15 11 68 77 37 13 46 23 27 6 35 50 80 2".split()]
    solution = [11, 2, 1, 23, 15, 19, 6, 5, 8, 9, 18, 24, 20, 21, 3, 12, 13, 4, 22, 17, 7, 10, 14, 16, 25]
    print(evaluate(solution, time_windows, travel_time ))

=======
>>>>>>> 1b339e424671cd19018724750f7d57b6e0aba514
