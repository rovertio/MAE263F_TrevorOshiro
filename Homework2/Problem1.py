import numpy as np
import matplotlib.pyplot as plt

from HelperFunctions.FrameFun import computeTangent, computeSpaceParallel, computeMaterialFrame, computeTimeParallel
from HelperFunctions.RodGeom import getRefTwist, getKappa
from HelperFunctions.ElasticFun import getFs, getFb, getFt
from HelperFunctions.PlottingFun import plotrod_simple



def objfun(qGuess, q0, u, a1, a2,
           freeIndex, # Boundary conditions
           dt, tol, # time stepping parameters
           refTwist, # We need a guess refTwist to compute the new refTwist
           massVector, mMat, # Mass vector and mass matrix
           EA, refLen, # Stretching stiffness and reference length\
           EI, GJ, voronoiRefLen, kappaBar, twistBar, # bending and twisting
           Fg):

  q = qGuess # Guess
  iter = 0
  error = 10 * tol

  while error > tol:
    a1Iterate, a2Iterate = computeTimeParallel(a1, q0, q) # Reference frame
    tangent = computeTangent(q)
    refTwist_iterate = getRefTwist(a1Iterate, tangent, refTwist) # Reference twist

    # Material frame
    theta = q[3::4] # twist angles
    m1Iterate, m2Iterate = computeMaterialFrame(a1Iterate, a2Iterate, theta)

    # Compute my elastic forces
    # Bending
    Fb, Jb = getFb(q, m1Iterate, m2Iterate, kappaBar, EI, voronoiRefLen) # Need to write this
    # Twisting
    Ft, Jt = getFt(q, refTwist_iterate, twistBar, GJ, voronoiRefLen) # Need to write this
    # Stretching
    Fs, Js = getFs(q, EA, refLen)

    # Set up EOMs
    Forces = Fb + Ft + Fs + Fg
    Jforces = Jb + Jt + Js
    f = massVector/dt * ( (q-q0)/dt - u ) - Forces
    J = mMat / dt**2 - Jforces
    # Free components of f and J to impose BCs
    f_free = f[freeIndex]
    J_free = J[np.ix_(freeIndex, freeIndex)]

    # Update
    dq_free = np.linalg.solve(J_free, f_free)

    q[freeIndex] = q[freeIndex] - dq_free # Update free DOFs
    error = np.sum(np.abs(f_free))

    print('Iter = %d' % iter)
    print('Error = %f' % error)

    iter += 1

  u = (q - q0) / dt # velocity vector

  return q, u, a1Iterate, a2Iterate


def mainloop(qGuess, q0, u, a1, a2, totalTime,
                freeIndex, # Boundary conditions
                dt, tol, # time stepping parameters
                refTwist, # We need a guess refTwist to compute the new refTwist
                massVector, mMat, # Mass vector and mass matrix
                EA, refLen, # Stretching stiffness and reference length\
                EI, GJ, voronoiRefLen, kappaBar, twistBar, # bending and twisting
                Fg):
    
    Nsteps = round(totalTime / dt )                     # Total number of steps
    time_array = np.arange(1, Nsteps + 1) * dt

    ctime = 0 # current time
    endZ = np.zeros(Nsteps) # Store z-coordinate of the last node with time

    for timeStep in range(Nsteps):
        print('Current time = %f' % ctime)

        qGuess = q0.copy() # This should be fixed - I did not include this line in class
        q, u, a1, a2 = objfun( qGuess, # Guess solution
                                q0, u, a1, a2,
                                freeIndex, # Boundary conditions
                                dt, tol, # time stepping parameters
                                refTwist, # We need a guess refTwist to compute the new refTwist
                                massVector, mMat, # Mass vector and mass matrix
                                EA, refLen, # Stretching stiffness and reference length
                                EI, GJ, voronoiRefLen, kappaBar, twistBar, # bending and twisting
                                Fg
                                )

        ctime += dt # Update current time

        # Update q0 with the new q
        q0 = q.copy()

        # Store the z-coordinate of the last node
        endZ[timeStep] = q[-1]

        # Every 100 time steps, update material directors and plot the rod
        if timeStep % 100 == 0:
            plotrod_simple(q, ctime)

    return time_array, endZ


def EndPlot(time_array, endZ):
    # Visualization after the loop
    plt.figure(2)
    plt.plot(time_array, endZ, 'ro-')
    plt.box(True)
    plt.xlabel('Time, t [sec]')
    plt.ylabel('z-coord of last node, $\\delta_z$ [m]')
    plot2_name = 'EndPlot'
    plt.savefig('Problem1_Plots/' + str(plot2_name))
    plt.show()


def main(totalTime, dt, nv):
    #################################################
    # Node coordinates and DoF initialization
    # nv = 50 # nodes
    ne = nv - 1 # edges
    ndof = 4 * nv - 1 # degrees of freedom: 3*nv + ne

    RodLength = 0.2 # meter
    natR = 0.02 # natural radius
    r0 = 0.001 # cross-sectional radius

    # Matrix (numpy ndarray) for the nodes at t=0
    nodes = np.zeros((nv, 3))
    if natR == 0: # straight rod
        for c in range(nv):
            nodes[c, 0] = c * RodLength / (nv - 1) # x coordinate of c-th node
            nodes[c, 1] = 0
            nodes[c, 2] = 0
    else: # rod with circular shape (ring)
        dTheta = (RodLength / natR) * (1.0 / ne)
        for c in range(nv):
            nodes[c, 0] = natR * np.cos((c-1) * dTheta)
            nodes[c, 1] = natR * np.sin((c-1) * dTheta)
            nodes[c, 2] = 0.0

    # DOF vector at t = 0
    q0 = np.zeros(ndof)
    for c in range(nv):
        ind = [4*c, 4*c+1, 4*c+2]
        q0[ind] = nodes[c, :]

    u = np.zeros_like(q0) # velocity vector

    #plotrod_simple(q0, 0)

    # Fixed and Free DOFs
    fixedIndex = np.arange(0,7) # First seven (2 nodes and one edge) are fixed: clamped
    freeIndex = np.arange(7,ndof)


    #################################################
    # Material / Physical Properties
    # Material parameters
    Y = 10e6 # Pascals
    nu = 0.5 # Poisson's raio
    G = Y / (2.0 * (1.0 + nu)) # shear modulus

    # Stiffness parameters
    EI = Y * np.pi * r0**4 / 4 # Bending stiffness
    GJ = G * np.pi * r0**4 / 2 # Twisting stiffness
    EA = Y * np.pi * r0**2 # Stretching stiffness

    rho = 1000 # Density (kg/m^3)
    totalM = (np.pi * r0**2 * RodLength) * rho # total mass in kg
    dm = totalM / ne # mass per edge

    massVector = np.zeros(ndof)
    for c in range(nv): # 0, 1, 2, ..., nv-1 MATLAB: for c=1:nv
        ind = [4*c, 4*c+1, 4*c+2]
        if c == 0: # first node
            massVector[ind] = dm/2
        elif c == nv-1: # last node
            massVector[ind] = dm/2
        else: # internal nodes
            massVector[ind] = dm

    for c in range(ne):
        massVector[4*c + 3] = 1/2 * dm * r0**2

    # Diagonal matrix rerpesentation of mass vector
    mMat = np.diag(massVector)

    # Gravity
    g = np.array([0, 0, -9.81])
    Fg = np.zeros(ndof) # External force vector for gravity
    for c in range(nv):
        ind = [4*c, 4*c+1, 4*c+2]
        Fg[ind] = massVector[ind] * g


    #################################################
    # Geometrical Properties

    # Reference (undeformed) length of each edge
    refLen = np.zeros(ne)
    for c in range(ne): # loop over each edge
        dx = nodes[c+1, :] - nodes[c, :] # edge vector from one node to the next
        refLen[c] = np.linalg.norm(dx)

    # Voronoi length of each node
    voronoiRefLen = np.zeros(nv)
    for c in range(nv): # loop over each node
        if c==0:
            voronoiRefLen[c] = 0.5 * refLen[c]
        elif c==nv-1:
            voronoiRefLen[c] = 0.5 * refLen[c-1]
        else:
            voronoiRefLen[c] = 0.5 * (refLen[c-1] + refLen[c])


    # Reference frame (Space parallel transport at t=0)
    a1 = np.zeros((ne,3)) # First reference director
    a2 = np.zeros((ne,3)) # Second reference director
    tangent = computeTangent(q0) # We need to create this function

    t0 = tangent[0,:] # tangent on the first edge
    t1 = np.array([0, 0, -1]) # "arbitrary" vector
    a1_first = np.cross(t0, t1) # This is perpendicular to tangent t0
    # Check for null vector
    if np.linalg.norm(a1_first) < 1e-6:
        t1 = np.array([0, 1, 0]) # new arbitrary vector
        a1_first = np.cross(t0, t1)
    a1_first = a1_first / np.linalg.norm(a1_first) # Normalize
    a1, a2 = computeSpaceParallel(a1_first, q0) # We need to create this function
    # a1, a2, tangent all have size (ne,3)

    # Material frame
    theta = q0[3::4] # twist angles
    m1, m2 = computeMaterialFrame(a1, a2, theta) # Compute material frame

    # Reference twist
    refTwist = np.zeros(nv)
    refTwist = getRefTwist(a1, tangent, refTwist) # We need to write this function

    # Natural curvature
    kappaBar = getKappa(q0, m1, m2) # We need to write this function

    # Natural twist
    twistBar = np.zeros(nv)


    #################################################
    # Simulation Execution
    tol = EI / RodLength**2 * 1e-3                      # Tolerance

    time_array, endZ = mainloop(q0, q0, u, a1, a2, totalTime,
                            freeIndex, # Boundary conditions
                            dt, tol, # time stepping parameters
                            refTwist, # We need a guess refTwist to compute the new refTwist
                            massVector, mMat, # Mass vector and mass matrix
                            EA, refLen, # Stretching stiffness and reference length\
                            EI, GJ, voronoiRefLen, kappaBar, twistBar, # bending and twisting
                            Fg)
    
    EndPlot(time_array, endZ)


if __name__ == "__main__":

    totalTime = float(input('Enter desired total time (Default for problem is 5s): ').strip() or '5')
    dt = float(input('Enter desired time step (Default for problem is 0.01s): ').strip() or '0.01')
    nv = int(input('Enter desired node number (Default for problem is 50): ').strip() or '50')

    main(totalTime, dt, nv)



