import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output # Only for IPython

# Helper Functions for MMM
# import HelperFunctions.collisions
import HelperFunctions.MMMadj as MMMadj
from HelperFunctions.BendingFun import getFbP1
from HelperFunctions.StrechingFun import getFsP1


def objfun(q_guess, q_old, u_old, dt, tol_dq, maximum_iter, m, mMat, W, free_index):
  q_new = q_guess.copy()
  # Newton-Raphson iterations
  iter_count = 0
  error = tol_dq * 10
  flag = 1 # Start with a good simulation

  while error > tol_dq:
    # Compute elastic forces
    # There is none
    # Equations of motion
    f = m / dt * ( (q_new - q_old) / dt - u_old ) - W
    # Jacobian
    J = mMat / dt**2.0
    # We only take the free part of f and J
    f_free = f[free_index]
    J_free = J[np.ix_(free_index, free_index)]

    # Solve for dq
    dq_free = np.linalg.solve(J_free, f_free)

    # Update q_new
    q_new[free_index] -= dq_free

    # Calculate error using the change in position (not force, but using force is OK too)
    error = np.linalg.norm(dq_free)
    iter_count += 1

    if iter_count > maximum_iter:
      flag = -1
      break

  reactionForce = f # f is zero (or close to it) for free DOFs. Non-zero at fixed DOFs.
  # The non-zero forces on fixed DOFs are my reaction forces.
  return q_new, flag, reactionForce


def simloop(q_guess, q_old, u_old, dt, mass, EI, EA, deltaL, force, tol, mat, nv):
    Nsteps = round(totalTime / dt) # number of time steps
    ctime = 0 # current time
    all_pos = np.zeros(Nsteps)
    q0 = q_old
    u = u_old
    all_pos[0] = q0[1]
    all_u = np.zeros(Nsteps)
    

    s_mat = np.eye(3*nv)
    z_vec = np.zeros(3*nv)

    for timeStep in range(1, Nsteps): # Loop over time steps
        print('t = %f\n' % ctime)
        r_force, q, flag = MMMadj.MMM_cal(q0, q0, u, dt, mass, EI, EA, deltaL, force, tol, s_mat, z_vec)
        #print("First")
        print(q)
        print(r_force)

        con_ind, free_ind, q_con, mat = MMMadj.test_col(q, r_force)
        print(con_ind)
        print(free_ind)

        if len(con_ind) < 2 and len(free_ind) < 2:
           print("error with collision detection")
           break
        # print(q_con)
        # print(mat[0])
        # print(mat[1])
        s_mat, z_vec = MMMadj.MMM_Szcalc(mat, con_ind, free_ind, q_con, q_old, u_old, dt, mass, force)
        print(s_mat)
        print(z_vec)

        r_force, q, flag = MMMadj.MMM_cal(q0, q0, u, dt, mass, EI, EA, deltaL, force, tol, s_mat, z_vec)
        print("Corrected")
        print(q[1])

        u = (q - q0) / dt # update velocity
        q0 = q.copy() # update old position

        all_pos[timeStep] = q0[1] # Save the positions
        all_u[timeStep] = u[1] # Save the positions
        ctime += dt # Update the current time

    return all_pos, u


def plotting(all_pos):
   # Plot
    plt.figure(1)
    plt.plot(all_pos[:,0], all_pos[:,1], 'ro', label='Node 1') # x,y plot for the first node
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    # plt.axis('equal')
    plt.xlim([-0.25, 0.25])
    plt.legend()
    plt.show()





if __name__ == '__main__':

    # Inputs
    nv = 2
    ndof = 3 * nv
    ne = nv - 1

    # Time step
    dt = 1e-4 # Play around with it to see it's aritificial effect on contact
    maximum_iter = 100
    totalTime = 3
    tol_dq = 1e-6 # Small length value

    # Mass
    # Rod Length
    RodLength = 0.10
    # Cross-sectional radius of rod
    r0 = 1e-3
    # Geometry of the rod
    nodes = np.zeros((nv, 2))
    for c in range(nv):
      nodes[c, 0] = c * RodLength / ne

    # Young's modulus
    Y = 1e9
    rho = 7000
    mass = 0.001

    # Discrete length
    deltaL = RodLength / (nv - 1)
    #deltaL=0.01
    # Radius of spheres
    R = np.zeros(nv)  # Vector of size N - Radius of N nodes
    R[:] = 0.005 # deltaL / 10: Course note uses deltaL/10
    # Utility quantities
    EI = Y * np.pi * r0**4 / 4
    EA = Y * np.pi * r0**2

    # Weight
    W = np.zeros(ndof)
    g = np.array([0, -9.8, 0])  # m/s^2 - gravity
    for k in range(nv):
      W[3*k]   = mass * g[0] # Weight for x_k
      W[3*k+1] = mass * g[1] # Weight for y_k

    # Initial conditions
    q0 = np.zeros(ndof)
    for k in range(nv):
      q0[3 * k] = nodes[c, 0]
      #q0[3 * k + 1] = nodes[c, 1]
      q0[3*k + 1] = 3

    # q0[1] = 3
    # q0[4] = 3
    # q0[7] = 3
    q = q0.copy()
    # Velocity
    u = np.zeros(ndof)
    # u[0:2] = [-0.005, 0]
    # u[2:4] = [0.005, 0]

    mat = np.zeros((nv,2,3))
    #mat = [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
    q_con = np.zeros(ndof)

    all_pos = simloop(q0, q0, u, dt, mass, EI, EA, deltaL, W, tol_dq, mat, nv)
    # plotting(all_pos)