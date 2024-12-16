from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def read_input(from_file=False, file_path=None):
    """
    Reads input for the TSP with Time Windows problem.
    
    Args:
        from_file (bool): Whether to read input from a file.
        file_path (str): Path to the input file (if from_file is True).

    Returns:
        N (int): Number of nodes (excluding the depot).
        time_windows (list of tuples): [(e0, l0, d0), (e1, l1, d1), ..., (eN, lN, dN)]
                                       0 is the depot.
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
            bool: True if the solution is valid, False otherwise.
            int: Total time taken for the route if valid, or -1 if invalid.
    """

    total_time = 0
    present_position = 0
    how_far = 0
    for next_position in solution:
        how_far += 1
        early_TW, late_TW, dur = time_windows[next_position]
        if next_position == 0: 
            total_time = max(total_time, early_TW) + dur  # Ready to go
            continue

        total_time += travel_time[present_position][next_position]
        total_time = max(total_time, early_TW)

        if total_time <= late_TW:   
            total_time += dur
        else: 
            print(f"Failed at step {how_far} - time window violated")
            return False, -1

        present_position = next_position

    # Add return to depot time
    total_time += travel_time[present_position][0]
    return True, total_time


def main():
    # Adjust according to your input source
    N, tws, t = read_input(from_file=True, file_path="TestCase/Subtask_100/N20ft301.dat")
    
    # tws[0] = (-1,-1,-1) for the depot.
    # Assign a large window and zero service time to the depot
    large_val = 10**9
    # Replace depot with large window and zero service time
    depot_window = (0, large_val, 0)
    tws[0] = depot_window
    
    # Extract customer data
    e = []
    l = []
    d = []
    for i in range(1, N+1):
        e_i, l_i, d_i = tws[i]
        e.append(int(e_i))
        l.append(int(l_i))
        d.append(int(d_i))
    
    data = {}
    data['N'] = N
    data['time_matrix'] = t
    data['time_windows'] = [(0, 100000)] + list(zip(e, l))  # Depot + customers
    data['service_time'] = [0] + d
    data['num_vehicles'] = 1
    data['depot'] = 0

    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'], 
                                           data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = data["time_matrix"][from_node][to_node]
        serv_time = data["service_time"][from_node]
        return int(travel_time + serv_time)  # cast to int if needed

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add time window constraints
    time_dim = "Time"
    routing.AddDimension(
        transit_callback_index,
        600000, 
        600000,  
        False,
        time_dim,
    )
    time_dimension = routing.GetDimensionOrDie(time_dim)
    
    # Set time windows for each node
    for location_idx, (e_win, l_win) in enumerate(data["time_windows"]):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(e_win, l_win)
    
    vehicle_id = 0
    # Minimize start and end times
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(vehicle_id)))
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(vehicle_id)))
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        # Extract the route from solution
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != data['depot']:
                route.append(node)
            index = solution.Value(routing.NextVar(index))
        
        # Construct full route with depot at start and end
        full_route = [0] + route + [0]

        # Evaluate the solution
        feasible, cost = evaluate(full_route, tws, t)
        print("N:", N)
        print("Route:", " ".join(map(str, route)))
        print("Feasible:", feasible)
        print("Cost:", cost)
    else:
        print(N)
        print("No solution found.")


if __name__ == "__main__":
    main()
