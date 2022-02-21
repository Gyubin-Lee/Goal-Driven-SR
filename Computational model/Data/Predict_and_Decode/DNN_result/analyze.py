import numpy as np
import csv
import math

def deque_mean_SE(deque):
    item_list = []
    while deque:
        item_list.append(deque.pop())
    item_array = np.array(item_list)
    item_array = item_array[:100]
    mean = np.mean(item_array)
    se = np.std(item_array)/math.sqrt(item_array.shape[0])
    
    return (mean, se)

TEST_RATE = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

for p in TEST_RATE:
    load_filename = 'result0120_p' + str(p) + '.npy'
    
    result = np.load(load_filename, allow_pickle = True)
    result2 = np.load(load_filename, allow_pickle = True)
    
    save_filename = 'result' + str(p)
    f = open(save_filename + '_mean.csv', 'w', newline = '')
    f2 = open(save_filename + '.csv', 'w', newline = '')
    wr = csv.writer(f)
    wr2 = csv.writer(f2)

    for sub in range(63):
        (mean, se) = deque_mean_SE(result[sub][0])
        wr.writerow([mean, se])
        list = []
        while result2[sub][0]:
            list.append(result2[sub][0].pop())
        wr2.writerow(list)