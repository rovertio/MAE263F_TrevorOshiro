import numpy as np
import matplotlib.pyplot as plt
from HelperFunctions.BendingFun import getFbP1
from HelperFunctions.StrechingFun import getFsP1


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
        Fb, Jb = getFbP1(q_new, EI, deltaL)
        Fs, Js = getFsP1(q_new, EA, deltaL)

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


def expMethod(q_guess, q_old, u_old, dt, tol, maximum_iter,
           m, mMat,  # inertia
           EI, EA,   # elastic stiffness
           W, C,     # external force
           deltaL):

        q_new = q_guess.copy()
        # Get elastic forces
        Fb, Jb = getFbP1(q_old, EI, deltaL)
        Fs, Js = getFsP1(q_old, EA, deltaL)

        # Viscous force
        # Fv = -C @ (q_new - q_old) / dt
        Fv = -C @ u_old
        # Jv = -C / dt
        
        # Explicit form: for each ball
        q_new = np.zeros(len(q_old))
        for i in range(3):
            x_ind = 2*i
            y_ind = 2*i + 1
            q_new[x_ind] = q_old[x_ind] + (dt**2/m[x_ind])*(((m[x_ind]/dt)*u_old[x_ind]) - (-Fb[x_ind] + -Fs[x_ind]) - (Fv[x_ind]))
            q_new[y_ind] = q_old[y_ind] + (dt**2/m[y_ind])*(((m[y_ind]/dt)*u_old[y_ind]) - (-Fb[y_ind] + -Fs[y_ind] + -W[y_ind]) - (Fv[y_ind]))

        # q_new = q_old + (dt)*((u_old) - (dt/m)*((-Fb + -Fs + -W) - Fv))
        print(q_new)
        
        return q_new 


def mainloop(sim, ctime, Nsteps, dt, q0, q, u,
             tol, maximum_iter,
             m, mMat,  # inertia
             EI, EA,   # elastic stiffness
             W, C,     # external force
             deltaL):

    all_pos = np.zeros(Nsteps)      # Tracks y position of R2
    all_v = np.zeros(Nsteps)        # Tracks y velocity of R2
    midAngle = np.zeros(Nsteps)     # Tracks mid angle at R2

    x1 = q[::2]  # Selects every second element starting from index 0
    x2 = q[1::2]  # Selects every second element starting from index 1
    h1 = plt.figure(1)
    plt.clf()  # Clear the current figure
    plt.plot(x1, x2, 'ko-')  # 'ko-' indicates black color with circle markers and solid lines
    plt.title('Structure deformation at time ' + str(0) + "s")  # Format the title with the current time
    plt.axis('equal')  # Set equal scaling
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plot1_name = str(sim) +'3nodeDeformation_0sec.png'
    plt.savefig('Homework1/Problem1_plots/' + str(plot1_name))

    for timeStep in range(1, Nsteps):  # Python uses 0-based indexing, hence range starts at 1
        #print(f't={ctime:.6f}')
        #print(timeStep)

        if sim == 'Implicit':
            q, error = objfun(q0, q0, u, dt, tol, maximum_iter, m, mMat, EI, EA, W, C, deltaL)

            if error < 0:
                print('Could not converge. Sorry')
                break  # Exit the loop if convergence fails
        else:
            q = expMethod(q0, q0, u, dt, tol, maximum_iter,
                            m, mMat,  # inertia
                            EI, EA,   # elastic stiffness
                            W, C,     # external force
                            deltaL)

        u = (q - q0) / dt  # velocity
        ctime += dt  # current time

        # Update q0
        q0 = q

        # rctime = np.round(ctime, 8)
        rctime = np.round(ctime, 3)
        print(rctime)

        if rctime == 0.01 or rctime == 0.05 or rctime == 0.10 or rctime == 1.00 or rctime == 10.00:
            x1 = q[::2]  # Selects every second element starting from index 0
            x2 = q[1::2]  # Selects every second element starting from index 1
            h1 = plt.figure(1)
            plt.clf()  # Clear the current figure
            plt.plot(x1, x2, 'ko-')  # 'ko-' indicates black color with circle markers and solid lines
            plt.title('Structure deformation at time ' + str(rctime) + "s")  # Format the title with the current time
            plt.axis('equal')  # Set equal scaling
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plot_name = str(sim) + '3nodeDeformation_{}sec.png'.format(f'{rctime}')
            plt.savefig('Homework1/Problem1_plots/' + str(plot_name))
            # plt.show()  # Display the figure
           


        all_pos[timeStep] = q[3]  # Python uses 0-based indexing
        all_v[timeStep] = u[3]

        # Angle at the center
        vec1 = np.array([q[2], q[3], 0]) - np.array([q[0], q[1], 0])
        vec2 = np.array([q[4], q[5], 0]) - np.array([q[2], q[3], 0])
        midAngle[timeStep] = np.degrees(np.arctan2(np.linalg.norm(np.cross(vec1, vec2)), np.dot(vec1, vec2)))

    print("Finished " + str(sim) + " method simulation")


    return all_pos, all_v, midAngle


def plotting(sim, totalTime, Nsteps, all_pos, all_v, midAngle):
    plt.figure(1)
    plt.clf() 
    t = np.linspace(0, totalTime, Nsteps)
    plt.plot(t, all_pos)
    plt.title(str(sim) + " method 3 sphere node position vs. time")
    plt.xlabel('Time, t [s]')
    plt.ylabel('Displacement, $\\delta$ [m]')
    plot1_name = str(sim) + '3nodeDisplacement.png'
    plt.savefig('Homework1/Problem1_plots/' + str(plot1_name))

    plt.figure(2)
    plt.plot(t, all_v)
    plt.title(str(sim) + " method 3 sphere node velocity vs. time")
    plt.xlabel('Time, t [s]')
    plt.ylabel('Velocity, v [m/s]')
    plot2_name = str(sim) + '3nodeVelocity.png'
    plt.savefig('Homework1/Problem1_plots/' + str(plot2_name))

    plt.figure(3)
    plt.plot(t, midAngle, 'r')
    plt.title(str(sim) + " method 3 sphere node mid-angle vs. time")
    plt.xlabel('Time, t [s]')
    plt.ylabel('Angle, $\\alpha$ [deg]')
    plot3_name = str(sim) + '3nodeMidAngle.png'
    plt.savefig('Homework1/Problem1_plots/' + str(plot3_name))


def main(met, R1, R2, R3, ex_dt, dt):

    # which methods to run
    flag_ex = np.zeros(2)
    if met == 'imp':
        flag_ex[1] = 1
    elif met == 'exp':
        flag_ex[0] = 1
    else:
        flag_ex = [1,1] 

    # Definition of Parameters: Inputs (SI units)
    nv = 3                          # number of vertices
    ne = nv - 1                     # number of edges
    midNode = int((nv + 1) / 2)

    # Rod Length
    RodLength = 0.10
    # Discrete length
    deltaL = RodLength / (nv - 1)
    # Cross-sectional radius of rod
    r0 = 1e-3
    # Young's modulus
    Y = 1e9

    # Densities
    rho_metal = 7000
    rho_gl = 1000
    rho = rho_metal - rho_gl

    # Viscosity
    visc = 1000.0

    # Utility quantities
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


    # Implicit solution execution
    if flag_ex[1] == 1:
        
        # Maximum number of iterations in Newton Solver
        maximum_iter = 100
        # Total simulation time (it exits after t=totalTime)
        totalTime = 10
        # Number of time steps
        Nsteps = round(totalTime / dt) + 1
        ctime = 0
        # Initial conditions
        q0 = np.zeros(2 * nv)
        for c in range(nv):
            q0[2 * c] = nodes[c, 0]
            q0[2 * c + 1] = nodes[c, 1]
        q = q0.copy()
        u = (q - q0) / dt

        all_pos, all_v, midAngle = mainloop("Implicit", ctime, Nsteps, dt, q0, q, u,
                    tol, maximum_iter,
                    m, mMat,  # inertia
                    EI, EA,   # elastic stiffness
                    W, C,     # external force
                    deltaL)

        plotting('Implicit', totalTime, Nsteps, all_pos, all_v, midAngle)

    # Explicit solution execution
    if flag_ex[0] == 1:
        
        # Maximum number of iterations in Newton Solver
        ex_maximum_iter = 100
        # Total simulation time (it exits after t=totalTime)
        ex_totalTime = 10
        # Number of time steps
        ex_Nsteps = round(ex_totalTime / ex_dt)
        ex_ctime = 0

        # Initial conditions
        ex_q0 = np.zeros(2 * nv)
        for c in range(nv):
            ex_q0[2 * c] = nodes[c, 0]
            ex_q0[2 * c + 1] = nodes[c, 1]

        ex_q = ex_q0.copy()
        ex_u = (ex_q - ex_q0) / ex_dt

        ex_all_pos, ex_all_v, ex_midAngle = mainloop("Explicit", ex_ctime, ex_Nsteps, ex_dt, ex_q0, ex_q, ex_u,
                    tol, ex_maximum_iter,
                    m, mMat,  # inertia
                    EI, EA,   # elastic stiffness
                    W, C,     # external force
                    deltaL)
        
        plotting('Explicit', ex_totalTime, ex_Nsteps, ex_all_pos, ex_all_v, ex_midAngle)


if __name__ == "__main__":

   in_R1 = float(input('Enter radii of left sphere (Default is 0.005m): ').strip() or '0.005')
   in_R2 = float(input('Enter radii of middle sphere (Default is 0.025m): ').strip() or '0.025')
   in_R3 = float(input('Enter radii of right sphere (Default is 0.005m): ').strip() or '0.005')
   in_met = str(input('Method used: exp for only explicit or imp for only implicit (Default is both): ').strip() or 'both')
   in_step_ex = float(input('Enter timestep of explicit method (Default is 1e-5): ').strip() or '0.00001')
   in_step_im = float(input('Enter timestep of explicit method (Default is 1e-2): ').strip() or '0.01')
   
   main(in_met, in_R1, in_R2, in_R3, in_step_ex, in_step_im)

   plt.show()



