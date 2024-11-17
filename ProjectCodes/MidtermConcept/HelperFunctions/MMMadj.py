import numpy as np


def get_matind(nodes):
    # takes in node numbers and outputs corresponding indices for S matrix
    # ASSUMING nodes start at 1
    ind = np.zeros(3*len(nodes))
    # print(ind)
    st_ind = 3 * (nodes - np.ones(len(nodes)))
    # print(st_ind)

    for ii in range(len(nodes)):
        ind[3*ii] = int(st_ind[ii])
        ind[3*ii + 1] = int(st_ind[ii] + 1)
        ind[3*ii + 2] = int(st_ind[ii] + 2)

    ind = ind.astype(int)

    return ind



def MMM_Scalc(mat, ndof, con_ind):
    # mat is a 3D array containing vectors p and q for a 2D case
    # nv is the total number of nodes in simulation
    # con_ind is the indices of nodes involved in collision
    # example call for two nodes: MMM_Scalc([[[1, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]], 6, [0,1,2,3,4,5])

    s_mat = np.eye(ndof)

    if ndof > 3:
        for ii in range(0, int(ndof/3) - 1):
            print(ii)
            S_n = np.eye(3) - np.outer(mat[ii][0], mat[ii][0]) - np.outer(mat[ii][1], mat[ii][1])
            print(ii)
            s_mat[(con_ind[3*ii]):(con_ind[3*ii] + 3), (con_ind[3*ii]):(con_ind[3*ii] + 3)] = S_n
            print(s_mat)
    else:
        # One node case
        S_n = np.eye(3) - np.outer(mat[0][0], mat[0][0]) - np.outer(mat[0][1], mat[0][1])
        s_mat[(con_ind[0]):(con_ind[0] + 3), (con_ind[0]):(con_ind[0] + 3)] = S_n

    return s_mat



def cor_cal(q_guess, q_old, u_old, dt, mass, p, q, q_con, force, nodes, tol):
    # Calculates the corrective placement of the node and the reaction force associated
    # p and q can be vectors with multiple entries of vectors

    # Obtaining S matrix and indicies for nodes under contact
    if len(np.shape(p)) == 1:
        mat = np.zeros([1,2])
    else:
        mat = np.zeros([np.size(p, 0), 2])
    mat[0:, 0] = p
    mat[0:, 1] = q
    cor_ind = get_matind(nodes)
    S_mat = MMM_Scalc(mat)
    

    return S_mat, cor_ind



def free_cal(q_guess, q_old, u_old, dt, mass, force, tol, ob_flag, mat, con_nodes, q_con, nv):
    # Calculates the placement of the node assuming no collision
    # q_con gives enforced displacement values at nodes from collision

    ndof = 3*nv
    # Simulation parameters
    q_new = q_guess.copy()
    iter_count = 0
    max_itt = 500
    error = tol * 10
    flag = 1 # Start with a good simulation

    # Check if collisions present and adjust accordingly
    con_ind = []
    uncon_ind = []
    if ob_flag == 1:
        con_ind = get_matind(con_nodes)                             # get indicies of constrained nodes
        # print(con_ind)
        uncon_ind = np.setdiff1d(np.arange(ndof), con_ind)          # get unconstrained nodes (not impacted from collision)

        S_mat = MMM_Scalc(mat, ndof, con_ind)                       # calculate S matrix
        q_new[con_ind] = q_con[con_ind]                             # set q_new for constrained nodes to set values
    else:
        S_mat = np.eye(ndof)                                        # set S matrix for all nodes unconstrained otherwise
        uncon_ind = np.arange(ndof)
    mMat = (1/mass)* S_mat


    while error > tol:
        # Calculation of correction step from mass matrix
        f_n = (1 / dt) * ( ((q_new - q_old) / dt) - u_old ) - (mMat @ force)
        # print(np.linalg.norm(f_n))
        # Summing all the constant value forces from constraints

        J_n = mMat / dt**2.0

        # Isolate particles unconstrained for updating in newton raphson
        # Constrained particles have fixed force for given q_new
        f_uncon = f_n[uncon_ind]
        J_uncon = J_n[np.ix_(uncon_ind, uncon_ind)]

        # Solve for dq
        dq_uncon = np.linalg.solve(J_uncon, f_uncon)
        # print(dq_uncon)
        q_new[uncon_ind] -= dq_uncon                            # Update q_new

        # Calculate error using the change in position (not force, but using force is OK too)
        error = np.linalg.norm(dq_uncon)

        print(iter_count)
        iter_count += 1
        if iter_count > max_itt:
            flag = -1
            break

    # Calculation of reaction force (done within iteration)
    r_force = f_n
    # Calculates projection of force onto the constraints if there are
    if len(con_ind) > 0:
        for jj in range(int(ndof/3) - 1):
            f_n[con_ind[(3*jj):(3*jj + 3)]] = ( np.dot(mat[jj][0], f_n[con_ind[(3*jj):(3*jj + 3)]]) ) * np.array(mat[jj][0])

    return r_force, q_new, flag



    

if __name__ == '__main__':
    
    nv = 1
    # nv = 2
    q_old = np.zeros(3)
    # q_old = np.zeros(6)
    q_guess = q_old.copy()
    u_old = np.zeros(3)
    # u_old = np.zeros(6)
    dt = 1e-3
    mass = 0.01
    force = mass * np.array([0, -9.81, 0])
    # force = mass * np.array([0, -9.81, 0, 0, -9.81, 0])
    tol = 1e-6
    ob_flag = -1
    mat = [[[0, 1, 0], [0, 0, 0]]]
    # mat = [[[0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
    con_nodes = [1]
    q_con = np.array([0, 0.001, 0])
    # q_con = np.array([0, 0.001, 0, 0, 0, 0]) 

    test_ind = get_matind([1])
    # print(type(test_ind))
    # print(test_ind[0])

    s_mat = MMM_Scalc(mat, 3, test_ind)
    # # print(mat[0][0])
    # print(s_mat)

    index = np.array(test_ind)
    # print(index)
    # print(q_con[index])

    # print(mat[0][0])
    # print(np.array(mat[0][0]) * np.dot(mat[0][0], [0.5, 0.5, 0.3]))
    
    r_force, q_new, flag = free_cal(q_guess, q_old, u_old, dt, mass, force, tol, ob_flag, mat, con_nodes, q_con, nv)
    print(r_force)
    print(q_new)
    print(flag)

