import numpy as np

from HelperFunctions.BendingFun import getFbP2
from HelperFunctions.StrechingFun import getFsP2
from HelperFunctions.OpFun import crossMat

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

def getNnum(ind):
    # takes in node indices and gives the corresponging number for 3D
    # ASSUMING nodes start at 1
    ind = np.array(ind)
    Nnum = np.zeros(int(len(ind)/3))
    st_nodes = np.ones(len(ind)) + ((ind/3).astype(int))

    for jj in range(int(len(ind)/3)):
        Nnum[jj] = st_nodes[3*jj]

    Nnum = Nnum.astype(int)

    return Nnum



def MMM_eq(q_new, q_old, u_old, dt, mass, force, S_mat, z_vec):
    # Iterates over nodes and adjusts according to collision presence

    # print(S_mat)
    #print(force)

    f_n = (1 / dt) * ( ((q_new - q_old) / dt) - u_old ) - ((1/mass)* S_mat @ force) - z_vec

    return f_n

def MMM_zcalc(q_con, q_old, u_old, dt, mass, force, S_mat):
    # Iterates over nodes and adjusts according to collision presence

    z_vec = (1 / dt) * ( ((q_con - q_old) / dt) - u_old ) - ((1/mass)* S_mat @ force)

    return z_vec



def MMM_Szcalc(mat, con_ind, free_ind, q_con, q_old, u_old, dt, mass, force):
    # mat is a 3D array containing vectors p and q for a 2D case
    # con_node -> indicates nodes that were free to be constricted by this step
    # free_node -> indicates nodes that were constrained to be freed in this step
    # example call for two nodes: MMM_Scalc([[[1, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]], 6, [0,1,2,3,4,5])

    ndof = len(q_old)                               # Degree of freedom from all nodes
    s_mat = np.eye(ndof)                            # Initialize the S matrix
    z_vec = np.zeros(ndof)                          # Initialize the z vector

    #con_node = getNnum(con_ind)
    #free_node = getNnum(free_ind)

    # Calculates the new S and z terms to correct in the objective function
    # If any nodes need to be constrained
    if len(con_ind) > 2:
        for ii in range(int(len(con_ind)/3)):
            S_n = np.eye(3) - np.outer(mat[ii][0], mat[ii][0]) - np.outer(mat[ii][1], mat[ii][1])
            z_n = MMM_zcalc(q_con[(con_ind[3*ii]):(con_ind[3*ii] + 3)], q_old[(con_ind[3*ii]):(con_ind[3*ii] + 3)], \
                            u_old[(con_ind[3*ii]):(con_ind[3*ii] + 3)], dt, mass, force[(con_ind[3*ii]):(con_ind[3*ii] + 3)], S_n)
            #print(z_n)
            z_n = ( np.dot(mat[ii][0], z_n) ) * np.array(mat[ii][0])

            s_mat[(con_ind[3*ii]):(con_ind[3*ii] + 3), (con_ind[3*ii]):(con_ind[3*ii] + 3)] = S_n
            z_vec[(con_ind[3*ii]):(con_ind[3*ii] + 3)] = z_n

    # If any nodes need to be freed
    if len(free_ind) > 2:
        for kk in range(int(len(free_ind)/3)):
            S_n = np.eye(3)
            z_n = np.zeros(3)

            s_mat[(free_ind[3*kk]):(free_ind[3*kk] + 3), (free_ind[3*kk]):(free_ind[3*kk] + 3)] = S_n
            z_vec[(free_ind[3*kk]):(free_ind[3*kk] + 3)] = z_n

            # s_mat[(con_ind[3*ii]):(con_ind[3*ii] + 3), (con_ind[3*ii]):(con_ind[3*ii] + 3)] = S_n
            # s_mat[(3*ii):(3*ii + 3), (3*ii):(3*ii + 3)] = S_n
            # z_vec[(3*ii):(3*ii + 3)] = z_n

    # print(s_mat)
    # print(z_vec)
    return s_mat, z_vec



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

        S_mat = MMM_Szcalc(mat, ndof, con_ind)                       # calculate S matrix
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



def test_col(q_test, r_force):
    # free_ind indicates nodes currently free that are being tested
    # con_ind indicates nodes currently constrained that are being tested
    q_con = q_test
    con_ind = np.zeros(len(q_test))
    con_ind = con_ind.astype(int)
    free_ind = np.zeros(len(q_test))
    free_ind = free_ind.astype(int)
    mat = np.zeros((int(len(q_test)/3),2,3))
    flag = 0

    # print(q_test)
    # print(r_force)

    for ii in range(int(len(q_test)/3)):
        # If the reaction force vector is zero
        if np.round(np.sum(r_force)) == 0:
            proj_vec = np.zeros(3)
        else:
            unit_r = (1/np.linalg.norm(r_force[(3*ii):(3*ii + 3)])) * r_force[(3*ii):(3*ii + 3)]
            proj_vec = ( np.dot(np.array([0,1,0]), unit_r) ) * np.array([0,1,0])

        #print(proj_vec[1])
        if q_test[3*ii + 1] < 1e-6:
            #q_con[3*ii + 1] = 0.1*(1-ii)
            q_con[3*ii + 1] = 0
            mat[ii] = np.array([[0,1,0],[0,0,0]])
            #print(mat)
            con_ind[ii] = ii + 1
            print("Below surface")
            flag = 1
        elif q_test[3*ii + 1] >= 1e-6 and proj_vec[1] < 0:
            free_ind[ii] = ii + 1
            mat[ii] = np.array([[0,0,0],[0,0,0]])
            print("Negative reaction")
        elif q_test[3*ii + 1] >= 1e-6 and np.round(proj_vec[1],8) == 0:
            # No reaction force -> free falling
            free_ind[ii] = ii + 1
            mat[ii] = np.array([[0,0,0],[0,0,0]])
            print("Zero force")
        else:
            free_ind[ii] = ii + 1
            mat[ii] = np.array([[0,0,0],[0,0,0]])
            print("Other")
            

    #print(free_ind)
    if len(con_ind[con_ind != 0]) >= 1:
        con_ind = get_matind(con_ind[con_ind != 0])
        #print(con_ind)
    else:
        con_ind = np.array([-1])
    if len(free_ind[free_ind != 0]) >= 1:
        # print(free_ind)
        free_ind = get_matind(free_ind[free_ind != 0])
        #print(free_ind)
    else:
        free_ind = np.array([-1])

    #print(mat)
    return con_ind, free_ind, q_con, mat, flag



def MMM_cal(q_guess, q_old, u_old, dt, mass, EI, EA, deltaL, force, tol, S_mat, z_vec):
    # Calculates the placement of the node assuming no collision
    # q_con gives enforced displacement values at nodes from collision

    # ndof = 3*nv
    # Simulation parameters
    q_new = q_guess.copy()
    iter_count = 0
    max_itt = 500
    error = tol * 10
    flag = 1 # Start with a good simulation

    while error > tol:
        # print(np.linalg.norm(f_n))
        # Fb, Jb = getFbP2(q_new, EI, deltaL)
        # Fs, Js = getFsP2(q_new, EA, deltaL)
        #print(Fb)
        #print(Fs)
        # print(Jb)
        # print(Js)

        # Calculation of correction step from mass matrix
        # For use with one node
        f_n = MMM_eq(q_new, q_old, u_old, dt, mass, (force), S_mat, z_vec)
        J_n = np.eye(len(q_old)) / dt**2.0
        # For use with 2 nodes
        # f_n = MMM_eq(q_new, q_old, u_old, dt, mass, (force+Fs), S_mat, z_vec)
        # J_n = np.eye(len(q_old)) / dt**2.0 + (Js)
        # For use with 3 nodes or more
        # f_n = MMM_eq(q_new, q_old, u_old, dt, mass, (force+Fs+Fb), S_mat, z_vec)
        # J_n = np.eye(len(q_old)) / dt**2.0 + (Js + Jb)

        
        # print(f_n)
        # print(J_n)
        # Solve for dq
        dq = np.linalg.solve(J_n, f_n)
        #print(dq)
        
        q_new = q_new - dq                            # Update q_new
        # Calculate error using the change in position (not force, but using force is OK too)
        error = np.linalg.norm(dq)

        # print(iter_count)
        iter_count += 1
        if iter_count > max_itt:
            flag = -1
            break
    #print(iter_count)
    # Calculation of reaction force (done within iteration)
    r_force = mass * f_n

    # Calculates projection of force onto the constraints if there are
    # if len(con_ind) > 0:
    #     for jj in range(int(ndof/3) - 1):
    #         f_n[con_ind[(3*jj):(3*jj + 3)]] = ( np.dot(mat[jj][0], f_n[con_ind[(3*jj):(3*jj + 3)]]) ) * np.array(mat[jj][0])

    return r_force, q_new, flag


    

if __name__ == '__main__':

    # a = [[1,2,3], [4, 5, 6], [7, 8, 9]]
    # print(a[np.ix_(np.array([1,2]), np.array([1,2]))])
    # print(getNnum(np.array([0,1,2,9,10,11,12,13,14])))

    nv = 2
    # nv = 2
    # q_old = np.zeros(3)
    q_old = np.zeros(6)
    q_guess = q_old.copy()
    # u_old = np.zeros(3)
    u_old = np.zeros(6)
    dt = 1e-3
    mass = 0.01
    # force = mass * np.array([0, -9.81, 0])
    force = mass * np.array([0, -9.81, 0, 0, -9.81, 0])
    tol = 1e-6
    ob_flag = -1
    # mat = [[[0, 1, 0], [0, 0, 0]]]
    mat = [[[0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
    # con_nodes = [1]
    #q_con = np.array([0, 0.001, 0])
    q_con = np.array([0, 0.001, 0, 0, 0, 0])
    con_ind = [0,1,2]
    free_ind = [3,4,5]
     

    test_ind = get_matind([1])
    # print(type(test_ind))
    # print(test_ind[0])

    # s_mat = MMM_Scalc(mat, 3, test_ind)
    # # print(mat[0][0])
    # print(s_mat)

    index = np.array(test_ind)
    # print(index)
    # print(q_con[index])

    # print(mat[0][0])
    # print(np.array(mat[0][0]) * np.dot(mat[0][0], [0.5, 0.5, 0.3]))
    
    S, z = MMM_Szcalc(mat, con_ind, free_ind, q_con, q_old, u_old, dt, mass, force)
    print(S)
    print(z)

