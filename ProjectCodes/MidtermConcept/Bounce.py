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
    all_rf = np.zeros(Nsteps)
    q0 = q_old
    q = q0
    u = u_old
    all_pos[0] = q0[1]
    all_u = np.zeros(Nsteps)

    maximum_iter=5
    
    r_force = np.zeros(3*nv)
    s_mat = np.eye(3*nv)
    z_vec = np.zeros(3*nv)

    for timeStep in range(1, Nsteps): # Loop over time steps
      print("--------------------------------------------------------------------")
      print('t = %f\n' % ctime)
      flag_c = 0
      # iter_count = 0

      #s_mat = np.eye(3*nv)
      #z_vec = np.zeros(3*nv)
      print(z_vec)
      r_force, q, flag = MMMadj.MMM_cal(q0, q0, u, dt, mass, EI, EA, deltaL, force, tol, s_mat, z_vec)
      print("Node position: " + str(q))
      print("Reaction force: " + str(r_force))
      con_ind, free_ind, q_con, mat, flag_c = MMMadj.test_col(q, r_force)
      print("Constraint nodes: " + str(con_ind))
      print("Free nodes: " + str(free_ind))
      s_mat, z_vec = MMMadj.MMM_Szcalc(mat, con_ind, free_ind, q_con, q0, u, dt, mass, force)
      print(s_mat)
      print(z_vec)

      if flag_c == 1:
        r_force, q, flag = MMMadj.MMM_cal(q0, q0, u, dt, mass, EI, EA, deltaL, force, tol, s_mat, z_vec)
        print("Node position: " + str(q))
        print("Reaction force: " + str(r_force))

      # while flag_c == 1:
        
      #   r_force, q, flag = MMMadj.MMM_cal(q0, q0, u, dt, mass, EI, EA, deltaL, force, tol, s_mat, z_vec)
      #   print("Node position: " + str(q))
      #   print("Reaction force: " + str(r_force))

      #   con_ind, free_ind, q_con, mat, flag_c = MMMadj.test_col(q, r_force)
      #   print("Constraint nodes: " + str(con_ind))
      #   print("Free nodes: " + str(free_ind))
      #   #print(q_con)

      #   if len(con_ind) < 2 and len(free_ind) < 2:
      #      print("error with collision detection")
      #      break

      #   s_mat, z_vec = MMMadj.MMM_Szcalc(mat, con_ind, free_ind, q_con, q0, u, dt, mass, force)
      #   print(s_mat)
      #   print(z_vec)

      #   # r_force, q, flag = MMMadj.MMM_cal(q0, q0, u, dt, mass, EI, EA, deltaL, force, tol, s_mat, z_vec)
      #   # print("Corrected")
      #   # print(q)

      #   iter_count += 1
      #   if iter_count > maximum_iter:
      #     print("excessive oscillation")
      #     break

      u = (q - q0) / dt # update velocity
      print("Velocity: " + str(u))
      q0 = q.copy() # update old position

      all_rf[timeStep] = r_force[1]
      all_pos[timeStep] = q0[1]
      # for jj in range(int(len(q0)/3)):
      #   all_pos[timeStep][0] = q0[1] # Save the positions
      #   all_pos[timeStep][1] = q0[1]
      all_u[timeStep] = u[1] # Save the positions
      ctime += dt # Update the current time

    return all_pos, all_rf, u


def plotting(all_pos, all_rf, totalTime, Nsteps):
    # Plot
    t = np.linspace(0, totalTime, Nsteps)
    # print(len(all_pos))
    # print(len(t))
    plt.figure(1)
    plt.plot(t, all_pos, 'ro', label='Node 1') # x,y plot for the first node
    plt.xlabel('t [s]')
    plt.ylabel('y [m]')
    # plt.axis('equal')
    #plt.xlim([-0.25, 0.25])
    plt.legend()

    plt.figure(2)
    plt.plot(t, all_rf, 'ro', label='Node 1') # x,y plot for the first node
    plt.xlabel('t [s]')
    plt.ylabel('rf [N]')
    # plt.axis('equal')
    #plt.xlim([-0.25, 0.25])
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # Mass
    RodLength = 0.10                          # Rod Length
    r0 = 1e-3                                 # Cross-sectional radius of rod

    #----------------------------------------------------------
    # Inputs for using one node
    # nv = 1
    # deltaL=0.01
    # ndof = 3 * nv
    # q0 = np.zeros(ndof)
    # q0[1] = 1
    # q = q0.copy() 

    #-----------------------------------------------------------
    # Inputs for using more than one node
    nv = 3
    ne = nv - 1
    deltaL = RodLength / (nv - 1)            # Discrete length
    ndof = 3 * nv

    nodes = np.zeros((nv, 2))
    for c in range(nv):
      nodes[c, 0] = c * deltaL
      nodes[c, 1] = 1
    q0 = np.zeros(ndof)
    for k in range(nv):
      q0[3 * k] = nodes[k, 0]
      q0[3 * k + 1] = nodes[k, 1]
      q0[3*k + 2] = 0
    print(q0)
    q = q0.copy()

    #------------------------------------------------------------


    # Time step
    dt = 1e-4 # Play around with it to see it's aritificial effect on contact
    maximum_iter = 100
    totalTime = 0.453
    #totalTime = 0.46
    Nsteps = round(totalTime / dt)
    tol_dq = 1e-6 # Small length value

    # Young's modulus
    Y = 1e9
    rho = 7000
    mass = 0.001

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
    
    # Velocity
    u = np.zeros(ndof)

    # Initialization of MMM parameters
    mat = np.zeros((nv,2,3))
    q_con = np.zeros(ndof)

    all_pos, all_rf, u = simloop(q0, q0, u, dt, mass, EI, EA, deltaL, W, tol_dq, mat, nv)
    plotting(all_pos, all_rf, totalTime, Nsteps)