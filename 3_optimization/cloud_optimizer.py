import argparse

from knapsack_solver import mc_knapsack

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("budget", type=int, \
        help="Total budget for the computation")
    parser.add_argument("-data", type=str, required=False, default='example_data.csv', \
        help="File that contains the input data")
    args = parser.parse_args()
    
    with open(args.data, 'r') as f:
        classes = list(map(int, f.readline().split(',')[1:]))
        runtime = list(map(int, map(float, f.readline().split(',')[1:])))
        cost_per_hour = list(map(float, f.readline().split(',')[1:]))
        cost = []
        values = []
        for i in range(len(runtime)):
            cost.append(runtime[i] * (cost_per_hour[i] / 60.0 / 60.0))
            values.append((runtime[i] * cost_per_hour[i] / 60.0 / 60.0))
        
        _, sol = mc_knapsack(args.budget, runtime, values, classes)
        if sol:
            total_runtime = 0
            total_cost = 0
            for k, v in sol.items():
                selected_machine = k*4 + v
                total_runtime += runtime[selected_machine]
                total_cost += cost_per_hour[selected_machine] * runtime[selected_machine] / 60 / 60
            
            print(sol)
            print('runtime: ', total_runtime / 60, total_runtime)
            print('cost: ', total_cost)
        else:
            print("NA")