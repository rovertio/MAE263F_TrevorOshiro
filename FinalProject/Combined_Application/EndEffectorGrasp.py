import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output # Only for IPython

# Helper Functions for MMM
# import HelperFunctions.collisions
import MMM_Combine as MMMadj
from HelperFunctions.BendingFun import getFbP1
from HelperFunctions.StrechingFun import getFsP1


def plot_contact(q, q_old, all_rfx, all_rfy, all_rfval, r_force, timeStep, ctime, xrec, yrec, x_obstacle, y_obstacle):

  # Obtain components of reaction force from the vector and compute magnitude
  for ii in range(int(len(q_old)/3)):
    all_rfx[timeStep][ii] = r_force[3*ii]
    all_rfy[timeStep][ii] = r_force[3*ii + 1]
    all_rfval[timeStep][ii] = np.sqrt((np.square(r_force[3*ii + 1]))+(np.square(r_force[3*ii])))

  if np.max(np.abs(all_rfval[timeStep])) == 0:    # If zero contact reaction forces present
    norm_rf = (-5)*np.ones(int(len(q_old)/3))
  else:                                           # If contact reaction forces present
    # Computation of reaction force magnitude
    norm_rf = ((np.abs(all_rfval[timeStep]) / np.max(np.abs(all_rfval[timeStep]))))

  x1 = q[0::3]                  # Selects every second element starting from index 0
  x2 = q[1::3]                  # Selects every second element starting from index 1

  plt.clf()                     # Clear the current figure

  # Plottint the obstacle
  #plt.plot(xrec, yrec)
  plt.plot(x_obstacle, y_obstacle, 'r-')
  plt.plot(x1, x2, 'k-')        

  # Plots nodes different colors depending on the relative magnitude of reaction force present
  for jj in range(int(len(q_old)/3)):
    if norm_rf[jj] == -5:
        plt.plot(x1[jj], x2[jj], 'o-', color=(0, 0, 1), markeredgewidth=1.5, markeredgecolor='black')
    else:
        plt.plot(x1[jj], x2[jj], 'o-', color=(np.abs((norm_rf[jj])), (1 - np.abs(norm_rf[jj])), 0), markeredgewidth=1.5, markeredgecolor='black')
  
  plt.title('time: ' + str(ctime))  # Format the title with the current time
  plt.xlim([-0.01, 0.011])
  plt.ylim([-0.08, -0.02])
  plt.xlabel('x [m]')
  plt.ylabel('y [m]')
  plot1b_name = 'ContactGripperGeom' + str(round(ctime, 5)) + '.png' 
  plt.savefig('Combined_Application/CombineApplicationPlots/Combine_ContactPlots/' + str(plot1b_name))
  plt.show()

  return all_rfx, all_rfy, all_rfval


# Execute function to begin simulation
def simloop(q_guess, q_old, u_old, dt, mass, EI, EA, deltaL, force, tol, mat, nv, fix_ix):
    # Simulation time values
    Nsteps = round(totalTime / dt)      # number of time steps
    ctime = 0                           # current time

    # Values to store/plot for forces
    all_rfx = np.zeros((Nsteps,(int(len(q_old)/3))))
    all_rfy = np.zeros((Nsteps,(int(len(q_old)/3))))
    all_rfval = np.zeros((Nsteps,(int(len(q_old)/3))))
    all_fin = np.zeros((Nsteps,(int(len(q_old)/3))))

    # Stored values of single nodes
    all_pos = np.zeros(Nsteps)
    all_zvec = np.zeros(Nsteps)
    all_u = np.zeros(Nsteps)

    # Initialize dispalcement and velocity values to compute
    q0 = q_old
    q = q0
    u = u_old
    all_pos[0] = q0[1]
    r_force = np.zeros(3*nv)

    # Initialize the Modified Mass method variables
    s_mat = np.eye(3*nv)
    z_vec = np.zeros(3*nv)

    # Variables controlling the adjustment of time-step in close proximity to surface
    dt_def = dt                   # Normal time step (1e-5)
    dt_c = 1e-6                   # Time step when close to object(1e-6)
    t_lastc = 0
    end_flag = 0
    close_d = 1e-5                # 1e-5
    close_off = 5e-6              # 5e-6

    # Coordinates for reference reactangle in contact 
    # Adjust test_col function in MMM_Combine.py to change the obstacle
    xrec, yrec = [-0.04, 0, 0], [-0.02, -0.02, -0.1]
    y_obstacle = np.linspace(-0.05 - 0.05, -0.05 + 0.05, 1000)
    x_obstacle = np.zeros(len(y_obstacle))
    for i in range(len(y_obstacle)):
      y_val = y_obstacle[i]
      x_obstacle[i] = MMMadj.right_circle(y_val, 0.05, -0.05, -0.05)
    

    for timeStep in range(1, Nsteps): # Loop over time steps
      print("--------------------------------------------------------------------------------------------- t = %f\n" % ctime)
      print("--------------------------------------------------------------------------------- x = %f\n" % (q0[0]) )
      print("-------------------------------------------------------------------- u = %f\n" % (u[0]))
      flag_c = 0                      # Collision flag

      # Predictor execution for displacement and force values
      r_force, f_in, q, flag = MMMadj.MMM_cal(q0, q0, u, dt_def, mass, EI, EA, deltaL, force, tol, s_mat, z_vec, mat, fix_ix)
      print("Node position: " + str(q))
      print("Reaction force: " + str(r_force))
      # Colission detection function to evaluate predictor
      con_ind, free_ind, q_con, mat, flag_c, close_flag = MMMadj.test_col(q, r_force, close_d, 0)
      print("Constraint nodes: " + str(con_ind))
      print("Free nodes: " + str(free_ind))

      # Loop over on a smaller time-step if the gripper is close to the surface
      if close_flag == 1:
        itt = 0        
        while close_flag == 1:
          print("close to contact")
          print("-------------------------------------------------------------------------- x = %f\n" % (q0[0]))
          print("------------------------------------------------------ u = %f\n" % (u[0]))
          s_mat, z_vec = MMMadj.MMM_Szcalc(mat, con_ind, free_ind, q_con, q0, u, dt_c, mass, force)
          r_force, f_in, q, flag = MMMadj.MMM_cal(q0, q0, u, dt_c, mass, EI, EA, deltaL, force, tol, s_mat, z_vec, mat, fix_ix)

          u = (q - q0) / dt_c                     # update velocity
          q0 = q.copy()                           # update old position
          con_ind, free_ind, q_con, mat, flag_c, close_flag = MMMadj.test_col(q, r_force, close_d, close_off)

          if flag_c == 1:
            # Calcualte new values for displacement and Modified Mass parameters when collision is detected
            s_mat, z_vec = MMMadj.MMM_Szcalc(mat, con_ind, free_ind, q_con, q0, u, dt_c, mass, force)
            print("Z vector: " + str(z_vec))
            r_force, f_in, q, flag = MMMadj.MMM_cal(q0, q0, u, dt_def, mass, EI, EA, deltaL, force, tol, s_mat, z_vec, mat, fix_ix)
            
            # Plot the collision
            #--------------------------------------------------------
            plot_contact(q, q_old, all_rfx, all_rfy, all_rfval, r_force, timeStep, ctime, xrec, yrec, x_obstacle, y_obstacle)
            #------------------------------------------------------

            # End simulation if excessive low amplitude oscillations
            # if timeStep - t_lastc < 200:
            #     end_flag = 1
            # t_lastc = timeStep
            # break
          itt += 1

          # Break out of the loop if a step of the larger time-step passes
          if itt == int(dt_def/dt_c):
              break
           
      else:
        s_mat, z_vec = MMMadj.MMM_Szcalc(mat, con_ind, free_ind, q_con, q0, u, dt_def, mass, force)
        if flag_c == 1:
          print("Z vector: " + str(z_vec))
          r_force, f_in, q, flag = MMMadj.MMM_cal(q0, q0, u, dt_def, mass, EI, EA, deltaL, force, tol, s_mat, z_vec, mat, fix_ix)
              
          # Avoid excessive plots over smaller time-steps
          #-----------------------------------------------
          if timeStep - t_lastc > 100:
            plot_contact(q, q_old, all_rfx, all_rfy, all_rfval, r_force, timeStep, ctime, xrec, yrec, x_obstacle, y_obstacle)
          #--------------------------------------------------------

          # End simulation if excessive low amplitude oscillations
          # if timeStep - t_lastc < 200:
          #   end_flag = 1
          #   #t_lastc = timeStep
          #   break

          t_lastc = timeStep
        
        u = (q - q0) / dt_def                     # update velocity
        print("Velocity: " + str(u))
        q0 = q.copy()                             # update old position

      # Storing internal forces for plotting
      for ii in range(int(len(q_old)/3)):
         all_fin[timeStep][ii] = np.sqrt((np.square(f_in[3*ii + 1]))+(np.square(f_in[3*ii])))
 
      # Storing displacement, velocity, and z vector values for plotting
      all_zvec[timeStep] = z_vec[6]
      all_pos[timeStep] = q0[6]
      all_u[timeStep] = u[6]                

      # Plot the positions over certain increments of time for visualization 
      if timeStep%700 == 0:
        x1 = q[0::3]                              # Selects every second element starting from index 0
        x2 = q[1::3]                              # Selects every second element starting from index 1
        plt.clf()                                 # Clear the current figure
        plt.plot(x1, x2, 'ko-')                   # 'ko-' indicates black color with circle markers and solid lines
        # plt.plot(xrec, yrec)
        plt.plot(x_obstacle, y_obstacle, 'r-')
        plt.title('time: ' + str(ctime))          # Format the title with the current time
        plt.xlim([-0.01, 0.011])
        plt.ylim([-0.11, 0.01])
        plt.axis('equal')                         # Set equal scaling
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plot1_name = 'Geom' + str(ctime) + '.png'
        plt.savefig('Combined_Application/CombineApplicationPlots/Combine_GeomPlots/' + str(plot1_name))
        plt.show()

      ctime += dt                                 # Update the current time
    

    return all_pos, all_u, all_fin, all_rfx, all_rfy, all_rfval, all_zvec, u


def plotting(all_pos, all_u, all_fin, all_rfx, all_rfy, all_rfval, all_zvec, totalTime, Nsteps):
    # Plotting for various properties of the simulation: displacement, velocity, reaction force

    # Time values to use for plotting
    t = np.linspace(0, totalTime, Nsteps)

    # Plotting horizontal displacement of selected node
    plt.figure(2)
    plt.clf()
    plt.plot(t, all_pos, 'ro', label='Node 1') # x,y plot for the first node
    plt.xlabel('t [s]')
    plt.ylabel('x [m]')
    plt.title("Horizontal discplacement of node 1")
    plt.legend()
    plot1_name = 'HorizontalDisplacementNode1.png'
    plt.savefig('Combined_Application/CombineApplicationPlots/' + str(plot1_name))

    # Plotting horizontal velocity of selected node
    plt.figure(3)
    plt.clf()
    plt.plot(t, all_u, 'ro', label='Node 1') # x,y plot for the first node
    plt.xlabel('t [s]')
    plt.ylabel('v [m/s]')
    plt.title("Horizontal velocity of node 1")
    plt.legend()
    plot2_name = 'HorizontalVelocityNode1.png'
    plt.savefig('Combined_Application/CombineApplicationPlots/' + str(plot2_name))

    # Plotting horizontal components node reaction forces
    plt.figure(4)
    plt.clf()
    for jj in range(int(np.size(all_rfx,1))):
      plt.plot(t, all_rfx[:,jj], 'o', label='Node ' + str(jj + 1)) # x,y plot for the first node
    plt.xlabel('t [s]')
    plt.ylabel('rf [N]')
    plt.title("Horizontal reaction force on nodes")
    plt.legend()
    plot3_name = 'HorizontalReactionForceNodes.png'
    plt.savefig('Combined_Application/CombineApplicationPlots/' + str(plot3_name))

    # Plotting vertical components of node reaction forces
    plt.figure(5)
    plt.clf()
    for jj in range(int(np.size(all_rfx,1))):
      plt.plot(t, all_rfy[:,jj], 'o', label='Node ' +  str(jj + 1))
    plt.xlabel('t [s]')
    plt.ylabel('rf [N]')
    plt.title("Vertical reaction force on nodes")
    plt.legend()
    plot4_name = 'VerticalReactionForceNodes.png'
    plt.savefig('Combined_Application/CombineApplicationPlots/' + str(plot4_name))

    # Magnitude of reaction force plotting
    plt.figure(6)
    plt.clf()
    for jj in range(int(np.size(all_rfx,1))):
      plt.plot(t, all_rfval[:,jj], 'o', label='Node ' +  str(jj + 1))
    plt.xlabel('t [s]')
    plt.ylabel('rf [N]')
    plt.title("Magnitude of reaction force on nodes")
    plt.legend()
    plot5_name = 'MagnitudeReactionForceNodes.png'
    plt.savefig('Combined_Application/CombineApplicationPlots/' + str(plot5_name))
    plt.show()


if __name__ == '__main__':

    #------------------------------------------------------------
    # Simulation execution properties
    # Time stepping
    dt = 1e-5                                 # Regular time-step 
    totalTime = 0.06                          # Total simulation run time
    Nsteps = round(totalTime / dt)            # Executions of simulation loop
    tol_dq = 1e-6                             # Small length value tolerance for objective function

    # Node number properties
    RodLength = 0.10                          # Rod Length
    r0 = 1e-3                                 # Cross-sectional radius of rod
    nv = 10
    ne = nv - 1
    deltaL = RodLength / (nv - 1)             # Discrete length
    ndof = 3 * nv

    # Applying boundary conditions to nodes
    all_DOFs = np.arange(ndof)
    fix_nodesNum = np.array([1,2])            # Nodes that are controlled with end-effector
    fix_nodes = max(fix_nodesNum)
    fix_ix = MMMadj.get_matind(fix_nodesNum)
    free_ix = np.setdiff1d(all_DOFs, fix_ix)

    #------------------------------------------------------------
    # Gripper prong geometry definition (displacement)

    # Straight line geometry
    # q0 = np.zeros(ndof)
    # for s in range(nv):
    #   q0[3 * s] = 0.01
    #   q0[3 * s + 1] = (-1) * s * deltaL
    #   q0[3*s + 2] = 0
    
    # Curved geometry
    q0 = np.zeros(ndof)
    for s in range(nv):
      q0[3 * s] = 0.001+0.8*np.square(0.1+((-1) * s * deltaL))
      q0[3 * s + 1] = (-1) * s * deltaL
      q0[3*s + 2] = 0

    #------------------------------------------------------------
    # Physical properties of gripper prong
    Y = 1e6                                   # Young's modulus                                
    mass = 0.01                               # Mass of discrete node

    # Utility quantities (cylindrical)
    # EI = Y * np.pi * r0**4 / 4
    # EA = Y * np.pi * r0**2
    # Utility quantities (rectangular)   
    b_len = 0.02                              # Reactangular CA width
    h_len = 0.01                              # Reactangular CA depth
    EI = Y * b_len * h_len**3 / 12
    EA = Y * b_len * h_len

    # Applied external forces: weight + actuator movement
    W = np.zeros(ndof)
    g = np.array([0, -9.8, 0])                # m/s^2 - gravity
    grip_acc = np.array([-9.8, 0, 0])
    for k in range(fix_nodes, nv):
      W[3*k+1] = mass * g[1]                  # Weight for y_k
    for p in range(fix_nodes):
      W[3*p]   = mass * grip_acc[0]           # Force in x direction

    #------------------------------------------------------------
    # Initilization of physcial simulation parameters
    # Position and velocity initialization
    q = q0.copy()
    u = np.zeros(ndof)

    # Initialization of MMM parameters
    mat = np.zeros((nv,2,3))
    q_con = np.zeros(ndof)
    #------------------------------------------------------------

    # Execution of the simulation loop and plotting function
    all_pos, all_u, all_fin, all_rfx, all_rfy, all_rfval, all_zvec, u = simloop(q0, q0, u, dt, mass, EI, EA, deltaL, W, tol_dq, mat, nv, fix_ix)
    plotting(all_pos, all_u, all_fin, all_rfx, all_rfy, all_rfval, all_zvec, totalTime, Nsteps)