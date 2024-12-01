import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output # Only for IPython

# Helper Functions for MMM
# import HelperFunctions.collisions
import MMM_BC as MMMadj
from HelperFunctions.BendingFun import getFbP1
from HelperFunctions.StrechingFun import getFsP1


def simloop(q_guess, q_old, u_old, dt, mass, EI, EA, deltaL, force, tol, mat, nv, free_ix):
    Nsteps = round(totalTime / dt) # number of time steps
    ctime = 0 # current time
    all_pos = np.zeros(Nsteps)
    all_rfx = np.zeros((Nsteps,(int(len(q_old)/3))))
    all_rfy = np.zeros((Nsteps,(int(len(q_old)/3))))
    all_zvec = np.zeros(Nsteps)
    all_u = np.zeros(Nsteps)
    coll_u = np.zeros(Nsteps)

    q0 = q_old
    q = q0
    u = u_old
    all_pos[0] = q0[1]

    dt_def = dt                   # 1e-5
    dt_c = 1e-6                   # 1e-6
    t_lastc = 0
    end_flag = 0
    close_d = 1e-5                # 1e-5
    close_off = 5e-6              # 5e-6

    # Coordinates for reference reactangle in contact
    xrec, yrec = [0.02, 0.02, 0.1], [-0.04, 0, 0]
    
    r_force = np.zeros(3*nv)
    s_mat = np.eye(3*nv)
    z_vec = np.zeros(3*nv)

    for timeStep in range(1, Nsteps): # Loop over time steps
      print("--------------------------------------------------------------------------------------------- t = %f\n" % ctime)
      print("--------------------------------------------------------------------------------- y = %f\n" % (q0[1]) )
      print("-------------------------------------------------------------------- u = %f\n" % (u[1]))
      #print('t = %f\n' % ctime)
      flag_c = 0

      #s_mat = np.eye(3*nv)
      #z_vec = np.zeros(3*nv)
      r_force, q, flag = MMMadj.MMM_cal(q0, q0, u, dt_def, mass, EI, EA, deltaL, force, tol, s_mat, z_vec, mat, free_ix)
      print("Node position: " + str(q))
      print("Reaction force: " + str(r_force))
      con_ind, free_ind, q_con, mat, flag_c, close_flag = MMMadj.test_col(q, r_force, close_d, 0)
      print("Constraint nodes: " + str(con_ind))
      print("Free nodes: " + str(free_ind))

      if close_flag == 1:
        itt = 0        
        #np.append(coll_u, np.zeros(int(dt_def/dt_c)))
        while close_flag == 1:
          print("close to contact")
          print("-------------------------------------------------------------------------- y = %f\n" % (q0[1]))
          print("------------------------------------------------------ u = %f\n" % (u[1]))
          s_mat, z_vec = MMMadj.MMM_Szcalc(mat, con_ind, free_ind, q_con, q0, u, dt_c, mass, force)

          r_force, q, flag = MMMadj.MMM_cal(q0, q0, u, dt_c, mass, EI, EA, deltaL, force, tol, s_mat, z_vec, mat, free_ix)

          u = (q - q0) / dt_c                     # update velocity
          q0 = q.copy()                         # update old position
          #coll_u[timeStep + itt] = u[1]

          con_ind, free_ind, q_con, mat, flag_c, close_flag = MMMadj.test_col(q, r_force, close_d, close_off)
          #print(close_flag)

          if flag_c == 1:
            s_mat, z_vec = MMMadj.MMM_Szcalc(mat, con_ind, free_ind, q_con, q0, u, dt_c, mass, force)
            print("Z vector: " + str(z_vec))
            r_force, q, flag = MMMadj.MMM_cal(q0, q0, u, dt_def, mass, EI, EA, deltaL, force, tol, s_mat, z_vec, mat, free_ix)
            
            # End simulation if excessive low amplitude oscillations
            if timeStep - t_lastc < 300:
                end_flag = 1
            t_lastc = timeStep
            break
          itt += 1

          if itt == int(dt_def/dt_c):
              break
        
      else:
        s_mat, z_vec = MMMadj.MMM_Szcalc(mat, con_ind, free_ind, q_con, q0, u, dt_def, mass, force)
        u = (q - q0) / dt_def                     # update velocity
        print("Velocity: " + str(u))
        q0 = q.copy()                         # update old position

      # Storing reaction forces for plotting
      for ii in range(int(len(q_old)/3)):
        all_rfx[timeStep][ii] = r_force[3*ii]
        all_rfy[timeStep][ii] = r_force[3*ii + 1]
      
      # Storing displacement, velocity, and z vector values for plotting
      all_zvec[timeStep] = z_vec[1]
      all_pos[timeStep] = q0[1]
      all_u[timeStep] = u[1]                # Save the positions

      #Break if excessive low oscillations
      # if end_flag == 1:
      #   print("terminated for oscillations")
      #   for ii in range(int(len(q_old)/3)):
      #     all_rfx[timeStep:-1][ii] = 0
      #     all_rfy[timeStep:-1][ii] = 0

      #   all_zvec[timeStep:-1] = 0
      #   all_pos[timeStep:-1] = 0
      #   all_u[timeStep:-1] = 0                # Save the positions
      #   break

      # Plot the positions
      if timeStep%500 == 0:
        x1 = q[0::3]  # Selects every second element starting from index 0
        #print(x1)
        x2 = q[1::3]  # Selects every second element starting from index 1
        #print(x2)
        plt.clf()  # Clear the current figure
        plt.plot(x1, x2, 'ko-')  # 'ko-' indicates black color with circle markers and solid lines
        plt.plot(xrec, yrec)
        plt.title('time: ' + str(ctime))  # Format the title with the current time
        plt.xlim([-0.1, 1.1])
        plt.axis('equal')  # Set equal scaling
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plot1_name = 'GripperGeom' + str(round(ctime, 3)) + '.png' 
        plt.savefig('FinalImplementation/BC_Application/BCApplicationPlots/BC_GeomPlots/' + str(plot1_name))
        # plt.show()

      ctime += dt # Update the current time
    
    coll_u = coll_u[coll_u != 0]
    # print(np.size(all_rfx[0,:]))
    # print(np.size(all_rfy[:,1]))

    return all_pos, all_u, all_rfx, all_rfy, all_zvec, coll_u, u


def plotting(all_pos, all_u, all_rfx, all_rfy, all_zvec, coll_u, totalTime, Nsteps):
    # Plot
    t = np.linspace(0, totalTime, Nsteps)
    # print(len(all_pos))
    # print(len(t))
    plt.figure(2)
    plt.clf()
    plt.plot(t, all_pos, 'ro', label='Node 1') # x,y plot for the first node
    plt.xlabel('t [s]')
    plt.ylabel('y [m]')
    plt.title("Vertical discplacement of node 1")
    plt.legend()
    plot1_name = 'VerticalDisplacementNode1.png'
    plt.savefig('FinalImplementation/BC_Application/BCApplicationPlots/' + str(plot1_name))

    plt.figure(3)
    plt.clf()
    plt.plot(t, all_u, 'ro', label='Node 1') # x,y plot for the first node
    plt.xlabel('t [s]')
    plt.ylabel('v [m/s]')
    plt.title("Vertical velocity of node 1")
    plt.legend()
    plot2_name = 'VerticalVelocityNode1.png'
    plt.savefig('FinalImplementation/BC_Application/BCApplicationPlots/' + str(plot2_name))

    plt.figure(4)
    plt.clf()
    for jj in range(int(np.size(all_rfx,1))):
      plt.plot(t, all_rfx[:,jj], 'o', label='Node ' + str(jj)) # x,y plot for the first node
    #plt.xlim([1.27, 1.34])
    plt.xlabel('t [s]')
    plt.ylabel('rf [N]')
    plt.title("Horizontal reaction force on node 1")
    plt.legend()
    plot3_name = 'HorizontalReactionForceNodes.png'
    plt.savefig('FinalImplementation/BC_Application/BCApplicationPlots/' + str(plot3_name))

    plt.figure(5)
    plt.clf()
    for jj in range(int(np.size(all_rfx,1))):
      plt.plot(t, all_rfy[:,jj], 'o', label='Node ' +  str(jj))
    #plt.xlim([1.27, 1.34])
    plt.xlabel('t [s]')
    plt.ylabel('rf [N]')
    plt.title("Vertical reaction force on node 1")
    plt.legend()
    plot4_name = 'VerticalReactionForceNodes.png'
    plt.savefig('FinalImplementation/BC_Application/BCApplicationPlots/' + str(plot4_name))

    # plt.figure(5)
    # plt.clf()
    # plt.plot(t, all_zvec, 'ro', label='Node 1') # x,y plot for the first node
    # #plt.xlim([1.27, 1.34])
    # plt.xlabel('t [s]')
    # plt.ylabel('a [m/s^2]')
    # plt.title("Vertical component of z vector")
    # plt.legend()
    # plot4_name = 'ZVectorNode1.png'
    # plt.savefig('FinalImplementation/BC_Application/BCApplicationPlots/' + str(plot4_name))
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
    nv = 26
    ne = nv - 1
    deltaL = RodLength / (nv - 1)            # Discrete length
    ndof = 3 * nv

    all_DOFs = np.arange(ndof)
    fixed_index = np.array([0, 1, 2])
    # Get the difference of two sets using np.setdiff1d
    free_ix = np.setdiff1d(all_DOFs, fixed_index)
    free_ix = fixed_index

    nodes = np.zeros((nv, 2))
    for c in range(nv):
      nodes[c, 0] = c * deltaL
      nodes[c, 1] = 0.01
    q0 = np.zeros(ndof)
    for k in range(nv):
      q0[3 * k] = nodes[k, 0]
      q0[3 * k + 1] = nodes[k, 1]
      q0[3*k + 2] = 0
    print(q0)
    q = q0.copy()

    #------------------------------------------------------------

    # Time step
    dt = 1e-5 # Play around with it to see it's aritificial effect on contact
    maximum_iter = 100
    #totalTime = 0.453
    #totalTime = 0.5
    totalTime = 0.07
    Nsteps = round(totalTime / dt)
    tol_dq = 1e-6 # Small length value

    # Young's modulus
    Y = 1e9
    rho = 7000
    mass = 0.01

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

    all_pos, all_u, all_rfx, all_rfy, all_zvec, coll_u, u = simloop(q0, q0, u, dt, mass, EI, EA, deltaL, W, tol_dq, mat, nv, free_ix)
    plotting(all_pos, all_u, all_rfx, all_rfy, all_zvec, coll_u, totalTime, Nsteps)