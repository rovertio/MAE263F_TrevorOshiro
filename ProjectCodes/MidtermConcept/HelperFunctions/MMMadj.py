import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output # Only for IPython

def get_matind(nodes):
    # takes in node numbers and outputs corresponding indices
    # ASSUMING nodes start at 1
    ind = np.zeros(3*len(nodes))
    st_ind = 3 * (nodes - np.ones(len(nodes)))

    for ii in range(len(nodes)):
        ind[3*ii] = st_ind[ii]
        ind[3*ii + 1] = st_ind[ii] + 1
        ind[3*ii + 2] = st_ind[ii] + 2

    return ind

def MMM_Scalc(mat):
    # mat is a 3D array containing vectors p and q for a 2D case
    # example call for two nodes: s_mat = MMM_Scalc([[[1, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]])

    if len(np.shape(mat)) == 1:
        Nnum = 1
    else:
        Nnum = np.size(mat, 0)

    s_mat = np.zeros([3*Nnum, 3*Nnum])
    # print(s_mat)

    for ii in range(Nnum):
        S_n = np.eye(3) - np.outer(mat[ii][0], mat[ii][0]) - np.outer(mat[ii][1], mat[ii][1])
        # print(ii)
        s_mat[(3*ii):(3*ii + 3), (3*ii):(3*ii + 3)] = S_n

    return s_mat



def cor_cal(q_new, q_old, u_old, dt, mass, p, q, q_con, force, nodes):
    # Calculates the corrective placement of the node and the reaction force associated
    # p and q can be vectors with multiple entries of vectors

    if len(np.shape(p)) == 1:
        mat = np.zeros([1,2])
    else:
        mat = np.zeros([np.size(p, 0), 2])
    
    mat[0:, 0] = p
    mat[0:, 1] = q

    cor_ind = get_matind(nodes)

    # Calculation of correction step from MMM
    S_mat = MMM_Scalc(mat)
    MMM_term = (1/mass)* S_mat @ force

    x_star = (1 / dt) * ( ((q_con - q_old) / dt) - u_old ) - MMM_term


    x_cor = dt*(u_old + x_star)
    q_new = x_cor + q_old

    # Calculation of reaction force
    r_force = (1 / dt) * ( ((q_new - q_old) / dt) - u_old ) - MMM_term

    return r_force, q_new




    

if __name__ == '__main__':
    # s_mat = MMM_Scalc([[[1, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]])
    # # s_mat = MMM_Scalc([[[1, 0, 0], [0, 0, 0]]])
    # print(s_mat)
    print(get_matind([1,3,4]))

