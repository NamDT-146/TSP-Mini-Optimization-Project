from ortools.linear_solver import pywraplp
from math import floor, ceil

def read_input(from_file=False, file_path=None):
    """
    Reads input for the TSP with Time Windows problem.
    Returns:
        N (int): number of customers (excluding depot),
                 so total nodes = N+1
        time_windows (list of tuples): [(e0, l0, d0), (e1, l1, d1), ..., (eN, lN, dN)]
        travel_time (list of lists): t(i,j) matrix of size (N+1)x(N+1)
    """
    if from_file and file_path:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        N = int(lines[0].strip())

        time_windows = [(-1, -1, -1)]
        for i in range(1, N + 1):
            e, l, d = map(float, lines[i].strip().split())
            time_windows.append((e, l, d))

        travel_time = []
        idx = N + 1
        for i in range(N + 1):
            row = list(map(float, lines[idx+i].strip().split()))
            travel_time.append(row)
    else:
        N = int(input().strip())
        time_windows = [(-1, -1, -1)]
        for _ in range(N):
            e, l, d = map(float, input().strip().split())
            time_windows.append((e, l, d))

        travel_time = []
        for _ in range(N + 1):
            row = list(map(float, input().strip().split()))
            travel_time.append(row)

    return N, time_windows, travel_time

def evaluate(solution, time_windows, travel_time):
    """
    Evaluates a given solution for the TSP with Time Windows problem.

    Args:
        solution (list): A route including depot at start and end, e.g. [0, ..., 0].
        time_windows (list): [(e(i), l(i), d(i)) for each node i].
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
        if next_position == 0 and how_far == 1:
            # Starting at depot
            total_time = max(total_time, early_TW) + dur
            continue
        total_time += travel_time[present_position][next_position]
        total_time = max(total_time, early_TW)

        if total_time <= late_TW:   
            total_time += dur
        else:
            print(f"Time window violated at step {how_far} for node {next_position}")
            return False, -1

        present_position = next_position

    return True, total_time

def solve_tsp_tw(N, time_windows, travel_time, t0=0):
    """
    Solves the TSP with Time Windows using a MIP model.
    
    Args:
        N (int): Number of customers (excluding depot).
        time_windows (list): A list of (e(i), l(i), d(i)) for i in [0..N], 
                             where 0 is the depot.
        travel_time (list): A (N+1)x(N+1) matrix of travel times.
        t0 (int/float): Start time at the depot.
    
    Returns:
        (route, cost): A tuple of (list_of_customers_visited_in_order, total_cost)
                       or (None, None) if no solution found.
    """
    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        print("Solver not found!")
        return None, None

    # x[i,j] binary variables
    x = {}
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                x[i,j] = solver.BoolVar(f'x_{i}_{j}')

    # Determine large M
    max_l = max(l for (e,l,d) in time_windows if l >= 0) # depot has -1 initially, skip those
    max_travel = max(tr for row in travel_time for tr in row)
    max_d = sum(d for (e,l,d) in time_windows if d >= 0)
    M = max_l + max_travel + max_d + 1000

    # T[i] for arrival time at node i
    T = [solver.NumVar(0, M, f'T_{i}') for i in range(N+1)]

    # Depot constraints
    solver.Add(sum(x[0,j] for j in range(1, N+1)) == 1)
    solver.Add(T[0] == t0)

    # Each customer visited once
    for i in range(1, N+1):
        solver.Add(sum(x[i,j] for j in range(N+1) if j != i) == 1)
        solver.Add(sum(x[k,i] for k in range(N+1) if k != i) == 1)

    # Time windows: redefine depot
    time_windows[0] = (0, M, 0) # large window, no service for depot
    for i in range(N+1):
        e_i, l_i, d_i = time_windows[i]
        solver.Add(T[i] >= e_i)
        solver.Add(T[i] <= l_i)

    # Time feasibility & subtour elimination
    for i in range(N+1):
        e_i, l_i, d_i = time_windows[i]
        for j in range(N+1):
            if i != j:
                solver.Add(T[j] >= T[i] + d_i + travel_time[i][j] - M*(1 - x[i,j]))

    # Minimize total travel time
    solver.Minimize(solver.Sum(travel_time[i][j]*x[i,j] for i,j in x))

    status = solver.Solve()
    if status == solver.OPTIMAL or status == solver.FEASIBLE:
        # Construct route
        route = []
        current = 0
        visited = set()
        while len(visited) < N:
            for j in range(N+1):
                if j != current and (current,j) in x and x[current,j].solution_value() > 0.5:
                    if j != 0:
                        route.append(j)
                        visited.add(j)
                    current = j
                    break
        cost = sum(travel_time[i][j]*x[i,j].solution_value() for i,j in x)
        return route, cost
    else:
        return None, None

def main():
    # Adjust input source as needed
    N, time_windows, travel_time = read_input(from_file=False, file_path=None)
    route, cost = solve_tsp_tw(N, time_windows, travel_time, t0=0)

    if route is not None:
        full_route = [0] + route + [0]
        feasible, eval_cost = evaluate(full_route, time_windows, travel_time)
        print(N)
        print(' '.join(map(str, route)))
        print("Feasible:", feasible)
        print("Cost according to MIP model:", cost)
        print("Cost according to evaluate function:", eval_cost)
    else:
        print(N)
        print(' '.join(map(str, range(1, N+1))))
        print("No solution found.")

if __name__ == "__main__":
    main()
