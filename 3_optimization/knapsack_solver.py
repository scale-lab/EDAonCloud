import sys

def knapsack(capacity, weights, values):
    K = [[0 for j in range(1 + capacity)] for i in range(len(values) + 1)]

    for i in range(len(values) + 1):
        for j in range(capacity + 1):
            if i == 0 or j == 0:
                K[i][j] = 0
            elif weights[i-1] <= j:
                K[i][j] = max(K[i-1][j], values[i-1] + K[i-1][j-weights[i-1]])
            else:
                K[i][j] = K[i-1][j]
    total_value = K[-1][-1]
    
    elements = []
    i = len(values)
    j = capacity
    while i != 0:
        if K[i][j] > K[i-1][j]:
            elements.append(i-1)
            j = j - weights[i-1]
        i = i-1
    elements.reverse()
    return total_value, elements


def mc_knapsack(capacity, weights, values, classes):
    KC = [[float('-inf') for j in range(1 + capacity)] for i in range(len(set(classes)) + 1)]

    items = list(zip(classes, weights, values))
    w = []
    p = []
    for i in range(len(set(classes))):
        w.append(list(list(zip(*list(filter(lambda x: x[0] == i, items))))[1]))
        p.append(list(list(zip(*list(filter(lambda x: x[0] == i, items))))[2]))

    selected = [[float('-inf') for j in range(1 + capacity)] for i in range(len(set(classes)) + 1)]
    for i in range(len(set(classes)) + 1):
        for j in range(capacity + 1):
            if i == 0:
                KC[i][j] = 0
            elif j == 0:
                KC[i][j] = float('-inf')
            else:
                choose_from = []
                indices = []
                for k in range(len(w[i-1])):
                    if 0 <= (j - w[i-1][k]):
                        choose_from.append(KC[i-1][j-w[i-1][k]] + p[i-1][k])
                        indices.append(k)
                if choose_from:
                    KC[i][j] = max(choose_from)
                    selected[i][j] = indices[choose_from.index(max(choose_from))]
                
    total_value = KC[-1][-1]

    # if no solution exists
    if total_value == float('-inf'):
        return None, None
    
    elements = {}
    i = len(set(classes))
    j = capacity
    while i != 0:
        elements[i-1] = selected[i][j]
        j = j - w[i-1][elements[i-1]]
        i = i-1
    return total_value, elements
    

if __name__ == "__main__":
    val = [120, 60, 100] 
    wt = [30, 10, 20] 
    W = 50
    print(knapsack(W, wt, val))
    print('----')
    val = [12, 6, 10, 18, 17, 20] 
    wt = [3, 2, 2, 4, 2, 5]
    c = [0, 0, 1, 1, 1, 2]
    W = 8
    print(mc_knapsack(W, wt, val, c))
    print('----')