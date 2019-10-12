import math

import numpy as np


def custom_function(x):
    x1 = x[0]
    x2 = x[1]
    return pow(x1 - 3, 2) + pow(x2 - 3, 2)


def testing_function(x1, x2):
    obj = 0.2 + x1 ** 2 + x2 ** 2 - 0.1 * math.cos(6.0 * 3.1415 * x1) - 0.1 * math.cos(6.0 * 3.1415 * x2)
    return obj


def testing_function2(x):
    obj = pow(x, 3) - 6 * pow(x, 2) - 15 * x + 100
    return obj


def testing_function3(x):
    obj = pow(x, 4) - 2*pow(x, 2) - 1
    return obj


def generate_normal_distribution_obj(best_previous_value, scatter_value, size_of_list, number_of_dimension):
    generated_obj = (np.random.normal(best_previous_value, scatter_value, size=[size_of_list, number_of_dimension]))
    # print(generated_list)
    return generated_obj


def acceptance_probability(cost_value, new_cost_value, actual_t):
    if new_cost_value < cost_value:
        return 1
    else:
        p = np.exp(- (new_cost_value - cost_value) / actual_t)
        return p


maxT = 10

decreaseT = 0.1

actualT = maxT

minT = 0.002
#
costValue = [0.0, 0.5]
# costValue = -3
# costValue = 0

values_difference = 0.5
amount_of_values = 1
# number_of_iterations = 35

dimension = 2

#
# while actualT > minT:
#     geneObj = generate_normal_distribution_obj(costValue, values_difference, amount_of_values)
#     if custom_function(geneObj[0, 0], geneObj[0, 1]) < custom_function(costValue[0], costValue[1]):
#         costValue[0] = geneObj[0, 0]
#         costValue[1] = geneObj[0, 1]
#     actualT = actualT - actualT * decreaseT
#

# fnc (x-3)^2
while actualT > minT:
    geneObj = generate_normal_distribution_obj(costValue, values_difference, amount_of_values, dimension)

    if acceptance_probability(custom_function(costValue),
                              custom_function(geneObj[0]), actualT) > np.random.random():
        costValue = geneObj[0]

    actualT = actualT - (actualT * decreaseT)
print(costValue, "with function val:", custom_function(costValue))

# fnc testing f 3
# while actualT > minT:
#     geneObj = generate_normal_distribution_obj(costValue, values_difference, amount_of_values, dimension)
#
#     if acceptance_probability(testing_function3(costValue),
#                               testing_function3(geneObj[0]), actualT) > np.random.random():
#         costValue = geneObj[0]
#
#     actualT = actualT - (actualT * decreaseT)

# print(costValue, "with function val:", testing_function3(costValue))
