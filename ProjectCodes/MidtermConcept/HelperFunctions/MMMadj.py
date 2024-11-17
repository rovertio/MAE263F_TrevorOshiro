import numpy as np


def get_matind(nodes):
    # takes in node numbers and outputs corresponding indices for S matrix
    # ASSUMING nodes start at 1
    ind = np.zeros(3*len(nodes))
    st_ind = 3 * (nodes - np.ones(len(nodes)))

    for ii in range(len(nodes)):
        ind[3*ii] = st_ind[ii]
        ind[3*ii + 1] = st_ind[ii] + 1
        ind[3*ii + 2] = st_ind[ii] + 2

    return ind



def MMM_Scalc(mat, ndof, con_ind):
    # mat is a 3D array containing vectors p and q for a 2D case
    # nv is the total number of nodes in simulation
    # con_ind is the indices of nodes involved in collision
    # example call for two nodes: MMM_Scalc([[[1, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]], 6, [0,1,2,3,4,5])

    s_mat = np.eye(ndof)
    # print(s_mat)

    for ii in range(len(con_ind/3)):
        S_n = np.eye(3) - np.outer(mat[ii][0], mat[ii][0]) - np.outer(mat[ii][1], mat[ii][1])
        # print(ii)
        s_mat[(con_ind(3*ii)):(con_ind(3*ii) + 3), (con_ind(3*ii)):(con_ind(3*ii) + 3)] = S_n

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
    max_itt = 100
    error = tol * 10
    flag = 1 # Start with a good simulation

    con_ind = []
    uncon_ind = []
    # Check if collisions present and adjust accordingly
    if ob_flag == 1:
        con_ind = get_matind(con_nodes)                             # get indicies of constrained nodes
        uncon_ind = np.setdiff1d(np.arange(ndof), con_ind)         # get unconstrained nodes (not impacted from collision)

        S_mat = MMM_Scalc(mat, ndof, con_ind)                       # calculate S matrix
        q_new[con_ind] = q_con[con_ind]                             # set q_new for constrained nodes to set values
    else:
        S_mat = np.eye(ndof)                                        # set S matrix for all nodes unconstrained otherwise
        uncon_ind = np.arange(ndof)
    mMat = (1/mass)* S_mat


    while error > tol:
        # Calculation of correction step from MMM
        f_n = (1 / dt) * ( ((q_new - q_old) / dt) - u_old ) - (mMat @ force)
        J_n = mMat / dt**2.0

        f_uncon = f_n[uncon_ind]
        J_uncon = J_n[np.ix_(uncon_ind, uncon_ind)]

        # Solve for dq
        dq_uncon = np.linalg.solve(J_uncon, f_uncon)
        # Update q_new
        q_new[uncon_ind] -= dq_uncon
        # Calculate error using the change in position (not force, but using force is OK too)
        error = np.linalg.norm(dq_uncon)

        iter_count += 1
        if iter_count > max_itt:
            flag = -1
            break

    # Calculation of reaction force (done within iteration)
    r_force = f_n
    # Calculates projection of force onto the constraints if there are
    if len(con_ind) > 0:
        for jj in range(len(con_ind/3)):
            f_n[con_ind[(3*jj):(3*jj + 3)]] = ( np.dot(mat[jj][0], f_n[con_ind[(3*jj):(3*jj + 3)]]) ) * mat[jj][0]

    return r_force, q_new, flag



    

if __name__ == '__main__':
    # s_mat = MMM_Scalc([[[1, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]])
    # # s_mat = MMM_Scalc([[[1, 0, 0], [0, 0, 0]]])
    # print(s_mat)
    #print(get_matind([1,3,4]))
    print(np.setdiff1d(np.arange(3), [0,1,2]))

