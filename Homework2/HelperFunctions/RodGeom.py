import numpy as np
from Operations import parallel_transport, rotateAxisAngle, signedAngle

# This file has the following functions: 
    # computeReferenceTwist(u1, u2, t1, t2, refTwist = None)
    # computekappa(node0, node1, node2, m1e, m2e, m1f, m2f)
    # getRefTwist(a1, tangent, refTwist)
    # getKappa(q0, m1, m2)



def computeReferenceTwist(u1, u2, t1, t2, refTwist = None):
    # This function computes the reference twist angle between two vectors "u1" and "u2",
    # given two tangent directions "t1" and "t2", and an optional initial guess for the twist.
    # It adjusts the guess to align "u1" with "u2" when parallel transported along "t1" and "t2".
    #
    # Parameters:
    #   u1: numpy array-like, shape (3,), the initial vector at position "t1".
    #   u2: numpy array-like, shape (3,), the target vector at position "t2".
    #   t1: numpy array-like, shape (3,), the initial tangent direction.
    #   t2: numpy array-like, shape (3,), the target tangent direction.
    #   refTwist: float, an optional initial guess for the reference twist angle (in radians).
    #             If not provided, it defaults to zero.
    #
    # Returns:
    #   refTwist: float, the adjusted reference twist angle (in radians) that aligns "u1"
    #             with "u2" after parallel transport and rotation.
    #
    # The function works by:
    # 1. Checking if "refTwist" is None and setting it to zero if no initial guess is provided.
    #    This ensures a default starting point for the twist calculation.
    # 2. Parallel transporting "u1" along "t1" to "t2" using the parallel_transport function.
    #    This adjusts "u1" so that it is aligned correctly in the direction of "t2".
    # 3. Rotating the parallel transported vector "ut" around "t2" by the current "refTwist" angle
    #    using the rotateAxisAngle function. This accounts for the initial twist guess.
    # 4. Computing the angle between the rotated "ut" and "u2" using the signedAngle function,
    #    with "t2" as the axis to determine the direction of the angle.
    # 5. Adjusting "refTwist" by adding the signed angle, thus refining the twist angle to
    #    align "u1" with "u2" after transport and rotation.
    #
    # Example:
    #   computeReferenceTwist(np.array([1, 0, 0]), np.array([0, 1, 0]),
    #                         np.array([0, 0, 1]), np.array([0, 0, 1]))
    #   This would compute the reference twist needed to align the x-axis with the y-axis
    #   after transporting along the same tangent direction, starting with an initial guess of 0.
    if refTwist is None:
      refTwist = 0
    ut = parallel_transport(u1, t1, t2)
    ut = rotateAxisAngle(ut, t2, refTwist)
    refTwist = refTwist + signedAngle(ut, u2, t2)
    return refTwist


def computekappa(node0, node1, node2, m1e, m2e, m1f, m2f):
    # This function computes the curvature "kappa" at a "turning" node in a discrete elastic rod model.
    # The curvature is calculated using the positions of three consecutive nodes and the material
    # directors of the edges before and after the turning point.
    #
    # Parameters:
    #   node0: array-like, shape (3,), the position of the node before the turning node.
    #          This represents the node before the bend in the rod.
    #   node1: array-like, shape (3,), the position of the "turning" node.
    #          This node is the point around which the curvature is calculated.
    #   node2: array-like, shape (3,), the position of the node after the turning node.
    #          This represents the node after the bend in the rod.
    #   m1e: array-like, shape (3,), material director 1 of the edge before the turning point.
    #        This is a vector that defines one direction in the material frame of the edge before turning.
    #   m2e: array-like, shape (3,), material director 2 of the edge before the turning point.
    #        This is a vector orthogonal to m1e in the material frame of the edge before turning.
    #   m1f: array-like, shape (3,), material director 1 of the edge after the turning point.
    #        This is a vector that defines one direction in the material frame of the edge after turning.
    #   m2f: array-like, shape (3,), material director 2 of the edge after the turning point.
    #        This is a vector orthogonal to m1f in the material frame of the edge after turning.
    #
    # Returns:
    #   kappa: array-like, shape (2,), the computed curvature at the turning node.
    #          - kappa[0]: Curvature component in the direction of the second material director.
    #          - kappa[1]: Curvature component in the direction of the first material director.
    #
    # The function works by:
    # 1. Computing the tangent vectors "t0" and "t1" for the edges before and after the turning node.
    #    - "t0" is the normalized vector from "node0" to "node1".
    #    - "t1" is the normalized vector from "node1" to "node2".
    # 2. Calculating the "curvature binormal" vector "kb", which measures the change in direction between
    #    "t0" and "t1". The formula ensures that the curvature is properly scaled and oriented.
    # 3. Initializing a "kappa" array to store the curvature components.
    # 4. Calculating "kappa1", which is proportional to the projection of "kb" onto the average of
    #    the material director "m2" vectors before and after the turning point.
    # 5. Calculating "kappa2", which is proportional to the negative projection of "kb" onto the average of
    #    the material director "m1" vectors before and after the turning point.
    # 6. Storing "kappa1" and "kappa2" in the "kappa" array and returning it.
    #
    # This function is crucial in simulating the bending behavior of a discrete elastic rod,
    # where the curvature at each turning point determines how the rod deforms. The use of
    # material directors allows for accurate representation of the bending directions in
    # the material frame.
    #
    # Example:
    #   computekappa(node0, node1, node2, m1e, m2e, m1f, m2f)
    #   This would compute the curvature for a given set of node positions and material directors.

    t0 = (node1 - node0) / np.linalg.norm(node1 - node0)
    t1 = (node2 - node1) / np.linalg.norm(node2 - node1)

    kb = 2.0 * np.cross(t0,t1) / (1.0 + np.dot(t0,t1))
    kappa1 = 0.5 * np.dot(kb,m2e + m2f)
    kappa2 = - 0.5 * np.dot(kb,m1e + m1f)

    kappa = np.zeros(2)
    kappa[0] = kappa1
    kappa[1] = kappa2

    return kappa


def getRefTwist(a1, tangent, refTwist):
  ne = a1.shape[0] # Shape of a1 is (ne,3)
  for c in np.arange(1,ne):
    u0 = a1[c-1,0:3] # reference frame vector of previous edge
    u1 = a1[c,0:3] # reference frame vector of current edge
    t0 = tangent[c-1,0:3] # tangent of previous edge
    t1 = tangent[c,0:3] # tangent of current edge
    refTwist[c] = computeReferenceTwist(u0, u1, t0, t1, refTwist[c])
  return refTwist


def getKappa(q0, m1, m2):
  ne = m1.shape[0] # Shape of m1 is (ne,3)
  nv = ne + 1

  kappa = np.zeros((nv,2))

  for c in np.arange(1,ne):
    node0 = q0[4*c-4:4*c-1]
    node1 = q0[4*c+0:4*c+3]
    node2 = q0[4*c+4:4*c+7]

    m1e = m1[c-1,0:3].flatten() # Material frame of previous edge
    m2e = m2[c-1,0:3].flatten() # NOT SURE if flattening is needed or not
    m1f = m1[c,0:3].flatten() # Material frame of current edge
    m2f = m2[c,0:3].flatten()

    kappa_local = computekappa(node0, node1, node2, m1e, m2e, m1f, m2f)

    # Store the values
    kappa[c,0] = kappa_local[0]
    kappa[c,1] = kappa_local[1]

  return kappa
