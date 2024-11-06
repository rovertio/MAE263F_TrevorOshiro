import numpy as np
from Operations import parallel_transport

# This file has the following functions: 
    # computeTangent(q)
    # computeSpaceParallel(d1_first, q)
    # computeMaterialFrame(a1, a2, theta)
    # computeTimeParallel(a1_old, q0, q)


def computeTangent(q):
  ne = int((len(q)+1)/4 - 1)
  tangent = np.zeros((ne, 3))
  for c in range(ne):
    dx = q[4*c+4:4*c+7] - q[4*c:4*c+3] # edge vector
    tangent[c,:] = dx / np.linalg.norm(dx) # make it unit
  return tangent


def computeSpaceParallel(d1_first, q):
  ne = int((len(q)+1)/4 - 1)
  tangent = computeTangent(q)

  d1 = np.zeros((ne, 3))
  d2 = np.zeros((ne, 3))

  # First edge
  d1[0,:] = d1_first # Given
  t0 = tangent[0,:] # Tangent on first edge
  d2[0,:] = np.cross(t0, d1_first)

  # Parallel transport from previous edge to the next
  for c in range(1, ne):
    t = tangent[c,:]
    d1_first = parallel_transport(d1_first, t0, t)
    # d1_first should be perpendicular to t
    d1_first = d1_first - np.dot(d1_first, t) * t
    d1_first = d1_first / np.linalg.norm(d1_first)

    # Store d1 and d2 vectors for c-th edge
    d1[c,:] = d1_first
    d2[c,:] = np.cross(t, d1_first) # I made a mistake in class and wrote cross(t0, d1_first)

    t0 = t.copy() # New tangent now becomes old tangent
    # I made a mistake in class and forgot to write "t.copy()" and just wrote "t0=t"

  return d1, d2


def computeMaterialFrame(a1, a2, theta):
  ne = len(theta)
  m1 = np.zeros((ne, 3))
  m2 = np.zeros((ne, 3))
  for c in range(ne): # loop over every edge
    m1[c,:] = a1[c,:] * np.cos(theta[c]) + a2[c,:] * np.sin(theta[c])
    m2[c,:] = - a1[c,:] * np.sin(theta[c]) + a2[c,:] * np.cos(theta[c])
  return m1, m2


def computeTimeParallel(a1_old, q0, q):
  # a1_old is (ne,3) ndarray representing old reference frame
  # q0 is the old DOF vector from where reference frame should be transported
  # q is the new DOF vector where reference frame should be transported to
  ne = int((len(q)+1)/4 - 1)
  tangent0 = computeTangent(q0) # Old tangents
  tangent = computeTangent(q) # New tangents

  a1 = np.zeros((ne, 3))
  a2 = np.zeros((ne, 3))
  for c in range(ne):
    t0 = tangent0[c,:]
    t = tangent[c,:]
    a1_tmp = parallel_transport(a1_old[c,:], t0, t)
    a1[c,:] = a1_tmp - np.dot(a1_tmp, t) * t
    a1[c,:] = a1[c,:] / np.linalg.norm(a1[c,:])
    a2[c,:] = np.cross(t, a1[c,:])

  return a1, a2