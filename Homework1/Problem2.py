import numpy as np
import matplotlib.pyplot as plt
from HelperFunctions.BendingFun import getFbP2
from HelperFunctions.StrechingFun import getFsP2

import numpy as np


def objfun(q_guess, q_old, u_old, dt, tol, maximum_iter,
           m, mMat,  # inertia
           EI, EA,   # elastic stiffness
           W, C,     # external force
           deltaL):

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
        Fv = -C @ (q_new - q_old) / dt
        Jv = -C / dt

        # Equation of motion
        f = m * (q_new - q_old) / dt**2 - m * u_old / dt - (Fb + Fs + W + Fv)

        # Manipulate the Jacobians
        J = mMat / dt**2 - (Jb + Js + Jv)

        # Newton's update
        q_new = q_new - np.linalg.solve(J, f)

        # Get the norm
        error = np.linalg.norm(f)

        # Update iteration number
        iter_count += 1
        # print(f'Iter={iter_count-1}, error={error:.6e}')

        if iter_count > maximum_iter:
            flag = -1  # return with an error signal
            return q_new, flag

    return q_new, flag


def mainloop(Nsteps, midNode, dt, q0, q, u,
             tol, maximum_iter,
             m, mMat,  # inertia
             EI, EA,   # elastic stiffness
             W, C,     # external force
             deltaL):

  ctime = 0

  all_pos = np.zeros(Nsteps)
  all_v = np.zeros(Nsteps)
  midAngle = np.zeros(Nsteps)

  for timeStep in range(1, Nsteps):  # Python uses 0-based indexing, hence range starts at 1
      print(f't={ctime:.6f}')

      q, error = objfun(q0, q0, u, dt, tol, maximum_iter, m, mMat, EI, EA, W, C, deltaL)

      if error < 0:
          print('Could not converge. Sorry')
          break  # Exit the loop if convergence fails

      u = (q - q0) / dt  # velocity
      ctime += dt  # current time

      # Update q0
      q0 = q

      all_pos[timeStep] = q[2*midNode-1]  # Python uses 0-based indexing
      all_v[timeStep] = u[2*midNode-1]

      # Angle at the center
      vec1 = np.array([q[2*midNode-2], q[2*midNode-1], 0]) - np.array([q[2*midNode-4], q[2*midNode-3], 0])
      vec2 = np.array([q[2*midNode], q[2*midNode+1], 0]) - np.array([q[2*midNode-2], q[2*midNode-1], 0])
      midAngle[timeStep] = np.degrees(np.arctan2(np.linalg.norm(np.cross(vec1, vec2)), np.dot(vec1, vec2)))

  return q, all_pos, all_v, midAngle


def plotting(totalTime, Nsteps, q, all_pos, all_v):

  # Plot
  x1 = q[::2]  # Selects every second element starting from index 0
  x2 = q[1::2]  # Selects every second element starting from index 1
  h1 = plt.figure(1)
  plt.clf()  # Clear the current figure
  plt.plot(x1, x2, 'ko-')  # 'ko-' indicates black color with circle markers and solid lines
  plt.title('Final deformed shape of 21 node falling beam')  # Format the title with the current time
  plt.axis('equal')  # Set equal scaling
  plt.xlabel('x [m]')
  plt.ylabel('y [m]')
  plot1_name = '21nodeDeformation' + str(totalTime) + 'sec.png'
  plt.savefig('Homework1/Problem2_plots/' + str(plot1_name))

  plt.figure(2)
  t = np.linspace(0, totalTime, Nsteps)
  plt.plot(t, all_pos)
  plt.title('Vertical displacement vs. time')
  plt.xlabel('Time, t [s]')
  plt.ylabel('Displacement, $\\delta$ [m]')
  plot2_name = '21nodeDisplacementTime' + str(totalTime) + 'sec.png'
  plt.savefig('Homework1/Problem2_plots/' + str(plot2_name))

  plt.figure(3)
  plt.plot(t, all_v)
  plt.plot(t[-1], all_v[-1], 'ko')
  plt.text(t[int(len(t)*0.6)], -0.0035, 't_vel: ' + str(round(all_v[-1], 5)) + ' [m/s]')
  plt.title('Vertical velocity vs. time')
  plt.xlabel('Time, t [s]')
  plt.ylabel('Velocity, v [m/s]')
  plot3_name = '21nodeVelocityTime' + str(totalTime) + 'sec.png'
  plt.savefig('Homework1/Problem2_plots/' + str(plot3_name))


def main(totalTime):
   
    # Inputs (SI units)
  # number of vertices
  nv = 21 # Odd vs even number should show different behavior
  ndof = 2*nv

  # Time step
  dt = 1e-2

  # Rod Length
  RodLength = 0.10

  # Discrete length
  deltaL = RodLength / (nv - 1)

  # Radius of spheres
  R = np.zeros(nv)  # Vector of size N - Radius of N nodes
  R[:] = 0.005 # deltaL / 10: Course note uses deltaL/10
  midNode = int((nv + 1) / 2)
  R[midNode -1 ] = 0.025

  # Densities
  rho_metal = 7000
  rho_gl = 1000
  rho = rho_metal - rho_gl

  # Cross-sectional radius of rod
  r0 = 1e-3

  # Young's modulus
  Y = 1e9

  # Viscosity
  visc = 1000.0

  # Maximum number of iterations in Newton Solver
  maximum_iter = 100

  # Indicate whether images should be saved
  saveImage = 0

  # Utility quantities
  ne = nv - 1
  EI = Y * np.pi * r0**4 / 4
  EA = Y * np.pi * r0**2

  # Tolerance on force function
  tol = EI / RodLength**2 * 1e-3  # small enough force that can be neglected

  # Geometry of the rod
  nodes = np.zeros((nv, 2))
  for c in range(nv):
      nodes[c, 0] = c * RodLength / ne

  # Compute Mass
  m = np.zeros(ndof)
  for k in range(nv):
    m[2*k] = 4 / 3 * np.pi * R[k]**3 * rho_metal # Mass for x_k
    m[2*k+1] = m[2*k] # Mass for y_k

  mMat = np.diag(m)  # Convert into a diagonal matrix

  # Gravity
  W = np.zeros(ndof)
  g = np.array([0, -9.8])  # m/s^2 - gravity
  for k in range(nv):
    W[2*k]   = m[2*k] * g[0] # Weight for x_k
    W[2*k+1] = m[2*k] * g[1] # Weight for y_k

  # Viscous damping matrix, C
  C = np.zeros((ndof, ndof))
  for k in range(nv):
    C[2*k,2*k]   = 6 * np.pi * visc * R[k]
    C[2*k+1, 2*k+1]   = 6 * np.pi * visc * R[k]

  # Initial conditions
  q0 = np.zeros(ndof)
  for c in range(nv):
      q0[2 * c] = nodes[c, 0]
      q0[2 * c + 1] = nodes[c, 1]

  q = q0.copy()
  u = (q - q0) / dt

  # Number of time steps
  Nsteps = round(totalTime / dt) + 1

  q, all_pos, all_v, midAngle = mainloop(Nsteps, midNode, dt, q0, q, u,
             tol, maximum_iter,
             m, mMat,  # inertia
             EI, EA,   # elastic stiffness
             W, C,     # external force
             deltaL)
  
  plotting(totalTime, Nsteps, q, all_pos, all_v)


if __name__ == "__main__":
   in_tot = float(input('Enter desired total time (Default for problem is 50s): ').strip() or '50')
   main(in_tot)

   plt.show()
