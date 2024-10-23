import numpy as np
import matplotlib.pyplot as plt
from P1BendingFun import getFb
from P1StrechingFun import getFs

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
        Fb, Jb = getFb(q_new, EI, deltaL)
        Fs, Js = getFs(q_new, EA, deltaL)

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
        print(f'Iter={iter_count-1}, error={error:.6e}')

        if iter_count > maximum_iter:
            flag = -1  # return with an error signal
            return q_new, flag

    return q_new, flag


# Definition of Parameters

# Inputs (SI units)
# number of vertices
nv = 3

# Time step
dt = 1e-2

# Rod Length
RodLength = 0.10

# Discrete length
deltaL = RodLength / (nv - 1)

# Radius of spheres
R1 = 0.005
R2 = 0.025
R3 = 0.005

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

# Total simulation time (it exits after t=totalTime)
totalTime = 10

# Indicate whether images should be saved
saveImage = 0

# How often the plot should be saved?
plotStep = 50

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
m = np.zeros(2 * nv)
m[0:2] = 4 / 3 * np.pi * R1**3 * rho_metal
m[2:4] = 4 / 3 * np.pi * R2**3 * rho_metal
m[4:6] = 4 / 3 * np.pi * R3**3 * rho_metal

mMat = np.diag(m)  # Convert into a diagonal matrix

# Gravity
W = np.zeros(2 * nv)
g = np.array([0, -9.8])  # m/s^2 - gravity
W[0:2] = 4 / 3 * np.pi * R1**3 * rho * g
W[2:4] = 4 / 3 * np.pi * R2**3 * rho * g
W[4:6] = 4 / 3 * np.pi * R3**3 * rho * g

# Viscous damping matrix, C
C = np.zeros((2 * nv, 2 * nv))
C1 = 6 * np.pi * visc * R1
C2 = 6 * np.pi * visc * R2
C3 = 6 * np.pi * visc * R3
C[0, 0] = C1
C[1, 1] = C1
C[2, 2] = C2
C[3, 3] = C2
C[4, 4] = C3
C[5, 5] = C3

# Initial conditions
q0 = np.zeros(2 * nv)
for c in range(nv):
    q0[2 * c] = nodes[c, 0]
    q0[2 * c + 1] = nodes[c, 1]

q = q0.copy()
u = (q - q0) / dt



# Execution of function
# Number of time steps
Nsteps = round(totalTime / dt)

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

    """
    if timeStep % plotStep == 0:
      x1 = q[::2]  # Selects every second element starting from index 0
      x2 = q[1::2]  # Selects every second element starting from index 1
      h1 = plt.figure(1)
      plt.clf()  # Clear the current figure
      clear_output(wait=True)  # Clear the previous plot/output: Only for iPython
      plt.plot(x1, x2, 'ko-')  # 'ko-' indicates black color with circle markers and solid lines
      plt.title(f't={ctime:.6f}')  # Format the title with the current time
      plt.axis('equal')  # Set equal scaling
      plt.xlabel('x [m]')
      plt.ylabel('y [m]')
      plt.show()  # Display the figure
    """

    all_pos[timeStep] = q[3]  # Python uses 0-based indexing
    all_v[timeStep] = u[3]

    # Angle at the center
    vec1 = np.array([q[2], q[3], 0]) - np.array([q[0], q[1], 0])
    vec2 = np.array([q[4], q[5], 0]) - np.array([q[2], q[3], 0])
    midAngle[timeStep] = np.degrees(np.arctan2(np.linalg.norm(np.cross(vec1, vec2)), np.dot(vec1, vec2)))

# Plot
plt.figure(2)
t = np.linspace(0, totalTime, Nsteps)
plt.plot(t, all_pos)
plt.xlabel('Time, t [s]')
plt.ylabel('Displacement, $\\delta$ [m]')
plt.savefig('fallingBeam.png')

plt.figure(3)
plt.plot(t, all_v)
plt.xlabel('Time, t [s]')
plt.ylabel('Velocity, v [m/s]')
plt.savefig('fallingBeam_velocity.png')

plt.figure(4)
plt.plot(t, midAngle, 'r')
plt.xlabel('Time, t [s]')
plt.ylabel('Angle, $\\alpha$ [deg]')
plt.savefig('fallingBeam_angle.png')

plt.show()