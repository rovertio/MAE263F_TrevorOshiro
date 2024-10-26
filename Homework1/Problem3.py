import numpy as np
import matplotlib.pyplot as plt
from HelperFunctions.BendingFun import getFbP2
from HelperFunctions.StrechingFun import getFsP2


def objfun(q_guess, q_old, u_old, dt, tol, maximum_iter,
           m, mMat,  # inertia
           EI, EA,   # elastic stiffness
           W, P,     # external force
           deltaL,
           free_index): # free_index indicates the DOFs that evolve under equations of motion

    q_new = q_guess.copy()

    # Newton-Raphson scheme
    iter_count = 0  # number of iterations
    error = tol * 10  # norm of function value (initialized to a value higher than tolerance)
    flag = 1  # Start with a 'good' simulation (flag=1 means no error)

    while error > tol:
        # Get elastic forces
        Fb, Jb = getFbP2(q_new, EI, deltaL)
        Fs, Js = getFsP2(q_new, EA, deltaL)

        # Viscous force
        # Fv = -C @ (q_new - q_old) / dt
        # Jv = -C / dt

        # Load force
        Fp = P

        # Equation of motion
        # f = m * (q_new - q_old) / dt**2 - m * u_old / dt - (Fb + Fs + W + Fv)
        f = m * (q_new - q_old) / dt**2 - m * u_old / dt - (Fb + Fs + Fp)

        # Manipulate the Jacobians
        # J = mMat / dt**2 - (Jb + Js + Jv)
        J = mMat / dt**2 - (Jb + Js)

        # We have to separate the "free" parts of f and J
        f_free = f[free_index]
        J_free = J[np.ix_(free_index, free_index)]

        # Newton's update
        # q_new = q_new - np.linalg.solve(J, f)
        # We have to only update the free DOFs
        dq_free = np.linalg.solve(J_free, f_free)
        q_new[free_index] = q_new[free_index] - dq_free

        # Get the norm
        # error = np.linalg.norm(f)
        # We have to calculate the errors based on free DOFs
        error = np.linalg.norm(f_free)

        # Update iteration number
        iter_count += 1
        # print(f'Iter={iter_count-1}, error={error:.6e}')

        if iter_count > maximum_iter:
            flag = -1  # return with an error signal
            return q_new, flag

    return q_new, flag

def mainloop(Nsteps, dt, q0, q, u,
             tol, maximum_iter,
             m, mMat,  # inertia
             EI, EA,   # elastic stiffness
             W, load,     # external force
             deltaL, RodLength,
             ndof, free_index):
  
  # Load Value
  P = np.zeros(ndof)
  P_app = 0.75                            # Load application location
  app_ind = int((P_app/RodLength)*ndof)
  if app_ind % 2 == 0:
      app_ind = app_ind - 1
  P[app_ind] = -load

  ctime = 0
  all_pos = np.zeros(Nsteps)

  for timeStep in range(1, Nsteps):  # Python uses 0-based indexing, hence range starts at 1
      
      if round(ctime,2)%0.1 == 0:
         print(f't={ctime:.6f}')

      q, error = objfun(q0, q0, u, dt, tol, maximum_iter,
            m, mMat,  # inertia
            EI, EA,   # elastic stiffness
            W, P,     # external force
            deltaL,
            free_index) # free_index indicates the DOFs that evolve under equations of motion

      if error < 0:
          print('Could not converge. Sorry')
          break  # Exit the loop if convergence fails

      u = (q - q0) / dt  # velocity
      ctime += dt  # current time

      # Update q0
      q0 = q

      all_pos[timeStep] = (-1) * np.max((-1)*q[1:len(q)-1:2])  # Python uses 0-based indexing
      # all_posmid[timeStep] = q[app_ind]  # Python uses 0-based indexing

  return q, all_pos

def EulerCalc(P, d, L, Y, I):
   c = np.min([d, (L-d)])
   y_max = (P*c*(L**2 - c**2)**1.5) / (9*np.sqrt(3)*Y*I*L)
   return y_max

def plottingDis(q, totalTime, load, Nsteps, all_pos, rel_load, q_max, eu_q_max):

  # Plot
  x1 = q[::2]  # Selects every second element starting from index 0
  x2 = q[1::2]  # Selects every second element starting from index 1
  h1 = plt.figure(1)
  plt.clf()  # Clear the current figure
  plt.plot(x1, x2, 'ko-')  # 'ko-' indicates black color with circle markers and solid lines
  plt.title('Final deformed shape of 50 node falling beam at time ' + 
            str(totalTime) + 's under ' + str(2000) + 'N load')  # Format the title with the current time
  plt.axis('equal')  # Set equal scaling
  plt.xlabel('x [m]')
  plt.ylabel('y [m]')
  plot1_name = '50nodeDeformation' + str(totalTime) + 'sec' + str(load) + 'Nload.png'
  plt.savefig('Homework1/Problem3_plots/' + str(plot1_name))

  plt.figure(2)
  t = np.linspace(0, totalTime, Nsteps)
  plt.plot(t, all_pos)
  plt.plot(t[-1], all_pos[-1], 'ko')
  plt.text(t[int(len(t)*0.6)], -0.034, 'y_max: ' + str(round(all_pos[-1], 5)) + ' [m]')
  plt.title('Maximum vertical displacement vs. time under ' + str(2000) + 'N load')
  plt.xlabel('Time, t [s]')
  plt.ylabel('Displacement, $\\delta$ [m]')
  plot2_name = '50nodeMaxY' + str(totalTime) + 'sec' + str(load) + 'Nload.png'
  plt.savefig('Homework1/Problem3_plots/' + str(plot2_name))

  plt.figure(3)
  plt.plot(rel_load, q_max, label = "Maximum Simulation Displacements")
  plt.plot(rel_load, eu_q_max, label = "Maximum Euler Displacements")
  plt.title('Comparison between Euler and simulation displacement values')
  plt.xlabel('Load, P [N]')
  plt.ylabel('Displacement, $\\delta$ [m]')
  plt.legend()
  plot3_name = '50nodeEulerComparison' + str(totalTime) + 'sec' + str(rel_load[-1]) + 'Maxload.png'
  plt.savefig('Homework1/Problem3_plots/' + str(plot3_name))

  plt.show()

def main(max_load, totalTime, dt, max_inc):
  # Inputs (SI units)
  # number of vertices
  nv = 50 # Odd vs even number should show different behavior
  ndof = 2*nv
  ne = nv - 1

  # Rod Length
  RodLength = 1
  # Discrete length
  deltaL = RodLength / (nv - 1)
  # Outer Cross-sectional radius of rod
  r0 = 0.013
  # Inner Cross-sectional radius of rod
  ri = 0.011

  # Geometry of the rod
  nodes = np.zeros((nv, 2))
  for c in range(nv):
      nodes[c, 0] = c * RodLength / ne

  
  rho_metal = 2700     # Densities
  Y = 70e9             # Young's modulus
  # Viscosity
  # visc = 1000.0
  # Stiffness quantities
  Inertia = np.pi * (r0**4 - ri**4) / 4
  EI = Y * Inertia
  EA = Y * np.pi * (r0**2 - ri**2)


  # Compute Mass
  m = np.zeros(ndof)
  for k in range(nv):
    # m[2*k] = 4 / 3 * np.pi * R[k]**3 * rho_metal # Mass for x_k
    m[2*k] = (np.pi * (r0**2 - ri**2) * RodLength * rho_metal) / (nv - 1) # Mass for x_k
    m[2*k+1] = m[2*k] # Mass for y_k
  mMat = np.diag(m)  # Convert into a diagonal matrix

  # Gravity
  W = np.zeros(ndof)
  g = np.array([0, -9.8])               # m/s^2 - gravity
  for k in range(nv):
    W[2*k]   = m[2*k] * g[0]            # Weight for x_k
    W[2*k+1] = m[2*k] * g[1]            # Weight for y_k
  # print(W)

  # Viscous damping matrix, C
  # C = np.zeros((ndof, ndof))
  # for k in range(nv):
  #   C[2*k,2*k]   = 6 * np.pi * visc * R[k]
  #   C[2*k+1, 2*k+1]   = 6 * np.pi * visc * R[k]

  # Load Value
  P = np.zeros(ndof)
  P_app = 0.75                            # Load application location
  load = 2000
  app_ind = int((P_app/RodLength)*ndof)
  if app_ind % 2 == 0:
      app_ind = app_ind - 1
  P[app_ind] = -load

  # Maximum number of iterations in Newton Solver
  maximum_iter = 100
  # Tolerance on force function
  tol = EI / RodLength**2 * 1e-3  # small enough force that can be neglected
  Nsteps = round(totalTime / dt) + 1             # Number of time steps

  # Initial conditions
  q0 = np.zeros(ndof)
  for c in range(nv):
      q0[2 * c] = nodes[c, 0]
      q0[2 * c + 1] = nodes[c, 1]

  q = q0.copy()
  u = (q - q0) / dt


  all_DOFs = np.arange(ndof)
  fixed_index = np.array([0, 1, 2*nv - 1])
  # Get the difference of two sets using np.setdiff1d
  free_index = np.setdiff1d(all_DOFs, fixed_index)

  q, all_pos = mainloop(Nsteps, dt, q0, q, u,
             tol, maximum_iter,
             m, mMat,  # inertia
             EI, EA,   # elastic stiffness
             W, 2000,     # external force
             deltaL, RodLength,
             ndof, free_index)
  print("Simulation with 2000N completed")

  
  q2, all_pos2 = mainloop(Nsteps, dt, q0, q, u,
             tol, maximum_iter,
             m, mMat,  # inertia
             EI, EA,   # elastic stiffness
             W, 20000,     # external force
             deltaL, RodLength,
             ndof, free_index)
  print("Simulation with 20000N completed")

  
  force_inc = max_load/max_inc        # increments of the force plot
  rel_load = force_inc*np.linspace(0, max_inc, max_inc + 1)
  q_max = np.zeros(max_inc + 1)
  eu_q_max = np.zeros(max_inc + 1)
  boff = np.zeros(max_inc + 1)
  

  for ii in range(max_inc + 1):
     plot_load = rel_load[ii]
     q_pl, all_pos_pl = mainloop(Nsteps, dt, q0, q, u,
             tol, maximum_iter,
             m, mMat,  # inertia
             EI, EA,   # elastic stiffness
             W, plot_load,     # external force
             deltaL, RodLength,
             ndof, free_index)
     q_max[ii] = -all_pos_pl[-1]
     
     qeu = EulerCalc(plot_load, 0.75, 1, Y, Inertia)
     eu_q_max[ii] = qeu

     if -(q_max[ii] - eu_q_max[ii]) > 0.005:
        boff[ii] = rel_load[ii]
     print("Calculations for load value {} finished".format(f"{plot_load}{"N"}"))
     
  # print(q_max)
  # print(eu_q_max)
  part1_ans = round(EulerCalc(2000, 0.75, 1, Y, Inertia), 3)
  part1_sim = round(all_pos[-1], 3)

  part2_ans = round(EulerCalc(20000, 0.75, 1, Y, Inertia), 3)
  part2_sim = round(all_pos2[-1], 3)

  print('Euler approximation at 2000N: ' + str(part1_ans) + 'm')
  print('Simulation displacement at 2000N: ' + str(part1_sim) + 'm')\
  
  print('Euler approximation at 20000N: ' + str(part2_ans) + 'm') 
  print('Simulation displacement at 20000N: ' + str(part2_sim) + 'm')

  # print(boff)
  print('Beam theory and simulation breaks off at ' + str(boff[np.min(np.nonzero(boff))]) + "N, with discrepancy > 0.01m")

  plottingDis(q, totalTime, load, Nsteps, all_pos, rel_load, q_max, eu_q_max)

if __name__ == "__main__":
   in_load = float(input('Enter maximum force for optional part (Default is 30000): ').strip() or '30000')
   in_inc = int(input('Enter increment amount for force for optional part (Default is 30): ').strip() or '30')
   in_time = float(input('Enter simulation time (Default is 1s): ').strip() or '1')
   in_step = float(input('Enter time step (Default is 0.01s): ').strip() or '0.005')

   main(in_load, in_time, in_step, in_inc)
