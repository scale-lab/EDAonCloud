from knapsack_solver import mc_knapsack

if __name__ == "__main__":
    with open('example_data.csv', 'r') as f:
        classes = list(map(int, f.readline().split(',')[1:]))
        runtime = list(map(int, map(float, f.readline().split(',')[1:])))
        cost_per_hour = list(map(float, f.readline().split(',')[1:]))
        cost = []
        values = []
        for i in range(len(runtime)):
            cost.append(runtime[i] * (cost_per_hour[i] / 60.0 / 60.0))
            values.append(1.0 / (runtime[i] * cost_per_hour[i] / 60.0 / 60.0))

        # sparc_core
        # budget = 5645 # 3, 3, 3, 3 = 128.88
        # budget = 1000  # no solution
        # budget = 6000 # -> 2, 2, 3, 1 = 267.02
        # budget = 10000  # -> 1, 0, 2, 1 = 274.61
        
        # coyote
        # budget = 5983 # 3, 3, 3, 3 = 160.09
        # budget = 1000 # no solution
        # budget = 8000

        # ariane
        # budget = 3500

        # swerv
        budget = 2100
        
        _, sol = mc_knapsack(budget, runtime, values, classes)
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