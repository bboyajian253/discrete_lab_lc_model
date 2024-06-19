import csv
import numpy as np
import my_toolbox as tb

import pars_shocks_and_wages as ps

def write_nd_array(writer, array, depth=0):
    if array.ndim == 1:
        writer.writerow(['  ' * depth + str(element) for element in array])
    else:
        for sub_array in array:
            write_nd_array(writer, sub_array, depth + 1)
            writer.writerow([])  # Blank row for separation at each level

def write_test():
    # Create a 3D array
    array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    # Write the array to a CSV file
    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        write_nd_array(writer, array)

def create_incr_arr_test():
    path = "STUFF"
    myPars = ps.Pars(path)
    my_arr = tb.create_increasing_array(myPars.state_space_shape_no_j)
    print(my_arr)

def gen_wage_coeff_grid_test():
    
    # w_coeff_grid = np.zeros([num_FE_types, 4])
    # for i in range(num_FE_types):
    #     w_coeff_grid[i, :] = [10.0*(i+1), 0.5*(i), -0.010*(i), 0.0]
    
    pass



if __name__ == '__main__':

    create_incr_arr_test()
