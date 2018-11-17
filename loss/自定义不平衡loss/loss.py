import numpy as np
def generate_weight_helper(data):
    count = np.zeros((4,1))
    count[0] = sum(data == -2)
    count[1] = sum(data == -1)
    count[2] = sum(data == 0)
    count[3] = sum(data == 1)
    return count

def generate_weight_1(data):
    count = generate_weight_helper(data)
    weight = np.sum(count) / count
    return (np.log(weight) / np.sum(np.log(weight))) * 4

def generate_weight_2(data):
    count = generate_weight_helper(data)
    count = np.log(count)
    weight = np.sum(count) / count
    return (np.log(weight) / np.sum(np.log(weight))) * 4
