import numpy as np
# This file has the following functions: 
    # signedAngle(u = None,v = None,n = None)
    # test_signedAngle()
    # rotateAxisAngle(v = None,z = None,theta = None)
    # test_rotateAxisAngle()
    # parallel_transport(u = None,t1 = None,t2 = None)
    # test_parallel_transport()
    # crossMat(a)



def signedAngle(u = None,v = None,n = None):
    # This function calculates the signed angle between two vectors, "u" and "v",
    # using an optional axis vector "n" to determine the direction of the angle.
    #
    # Parameters:
    #   u: numpy array-like, shape (3,), the first vector.
    #   v: numpy array-like, shape (3,), the second vector.
    #   n: numpy array-like, shape (3,), the axis vector that defines the plane
    #      in which the angle is measured. It determines the sign of the angle.
    #
    # Returns:
    #   angle: float, the signed angle (in radians) from vector "u" to vector "v".
    #          The angle is positive if the rotation from "u" to "v" follows
    #          the right-hand rule with respect to the axis "n", and negative otherwise.
    #
    # The function works by:
    # 1. Computing the cross product "w" of "u" and "v" to find the vector orthogonal
    #    to both "u" and "v".
    # 2. Calculating the angle between "u" and "v" using the arctan2 function, which
    #    returns the angle based on the norm of "w" (magnitude of the cross product)
    #    and the dot product of "u" and "v".
    # 3. Using the dot product of "n" and "w" to determine the sign of the angle.
    #    If this dot product is negative, the angle is adjusted to be negative.
    #
    # Example:
    #   signedAngle(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    #   This would return a positive angle (π/2 radians), as the rotation
    #   from the x-axis to the y-axis is counterclockwise when viewed along the z-axis.
    w = np.cross(u,v)
    angle = np.arctan2( np.linalg.norm(w), np.dot(u,v) )
    if (np.dot(n,w) < 0):
        angle = - angle

    return angle


def test_signedAngle():
  """
  This function tests the signedAngle function with three test cases.
  """
  # Test case 1: Orthogonal vectors
  u = np.array([1, 0, 0])
  v = np.array([0, 1, 0])
  n = np.array([0, 0, 1])
  angle = signedAngle(u, v, n)
  assert np.isclose(angle, np.pi/2), "Test case 1 failed"

  # Test case 2: Parallel vectors
  u = np.array([1, 1, 1])
  v = np.array([2, 2, 2])
  n = np.array([0, 1, 0])
  angle = signedAngle(u, v, n)
  assert np.isclose(angle, 0), "Test case 2 failed"

  # Test case 3: Anti-parallel vectors
  u = np.array([1, 1, 1])
  v = np.array([-1, -1, -1])
  n = np.array([0, 1, 0])
  angle = signedAngle(u, v, n)
  assert np.isclose(angle, np.pi), "Test case 3 failed"

  print("All test cases passed")


def rotateAxisAngle(v = None,z = None,theta = None):
    # This function rotates a vector "v" around a specified axis "z" by an angle "theta".
    #
    # Parameters:
    #   v: numpy array-like, shape (3,), the vector to be rotated.
    #   z: numpy array-like, shape (3,), the unit vector representing the axis of rotation.
    #      It should be normalized before calling the function for correct results.
    #   theta: float, the rotation angle in radians.
    #
    # Returns:
    #   vNew: numpy array-like, shape (3,), the rotated vector after applying the rotation
    #         around axis "z" by angle "theta".
    #
    # The function works by:
    # 1. Checking if the rotation angle "theta" is zero. If so, the function returns the original
    #    vector "v" unchanged since no rotation is needed.
    # 2. If "theta" is not zero, it computes the new rotated vector using the formula:
    #      vNew = cos(theta) * v + sin(theta) * (z × v) + (1 - cos(theta)) * (z · v) * z
    #    This formula is derived from Rodrigues' rotation formula, which calculates the rotation
    #    of a vector around an axis using trigonometric functions and vector operations.
    #    - The term cos(theta) * v represents the component of "v" that remains aligned with "v".
    #    - The term sin(theta) * (z × v) gives the component perpendicular to both "z" and "v".
    #    - The term (1 - cos(theta)) * (z · v) * z adjusts the result based on the projection of "v"
    #      onto the axis "z".
    #
    # Example:
    #   rotateAxisAngle(np.array([1, 0, 0]), np.array([0, 0, 1]), np.pi/2)
    #   This would rotate the vector [1, 0, 0] by 90 degrees around the z-axis, returning
    #   a vector close to [0, 1, 0].
    if (theta == 0):
        vNew = v
    else:
        c = np.cos(theta)
        s = np.sin(theta)
        vNew = c * v + s * np.cross(z,v) + np.dot(z,v) * (1.0 - c) * z

    return vNew


def test_rotateAxisAngle():
  """
  This function tests the rotateAxisAngle function by comparing the output with
  the expected output for a given set of inputs.
  """

  # Test case: Rotate a vector by 90 degrees around the z-axis
  v = np.array([1, 0, 0])
  axis = np.array([0, 0, 1])
  theta = np.pi/2
  v_rotated = rotateAxisAngle(v, axis, theta)

  # Expected output
  v_expected = np.array([0, 1, 0])

  # Check if the output is close to the expected output
  assert np.allclose(v_rotated, v_expected), "Test case failed"

  print("Test case passed")


def parallel_transport(u = None,t1 = None,t2 = None):

    # This function parallel transports a vector u from tangent t1 to t2
    # Input:
    # t1 - vector denoting the first tangent
    # t2 - vector denoting the second tangent
    # u - vector that needs to be parallel transported
    # Output:
    # d - vector after parallel transport

    b = np.cross(t1,t2)
    if (np.linalg.norm(b) == 0):
        d = u
    else:
        b = b / np.linalg.norm(b)
        # The following four lines may seem unnecessary but can sometimes help
        # with numerical stability
        b = b - np.dot(b,t1) * t1
        b = b / np.linalg.norm(b)
        b = b - np.dot(b,t2) * t2
        b = b / np.linalg.norm(b)
        n1 = np.cross(t1,b)
        n2 = np.cross(t2,b)
        d = np.dot(u,t1) * t2 + np.dot(u,n1) * n2 + np.dot(u,b) * b

    return d


def test_parallel_transport():
  """
  This function tests the parallel_transport function by checking if the
  transported vector is orthogonal to the new tangent vector.
  """

  # Test case 1: Orthogonal tangents
  u = np.array([1, 0, 0])
  t1 = np.array([0, 1, 0])
  t2 = np.array([0, 0, 1])
  u_transported = parallel_transport(u, t1, t2)
  assert np.allclose(np.dot(u_transported, t2), 0), "Test case 1 failed"
  # Returns True if two arrays are element-wise equal within a tolerance.

  # Test case 2: Parallel tangents
  u = np.array([1, 1, 1])
  t1 = np.array([1, 0, 0])
  t2 = np.array([2, 0, 0])
  u_transported = parallel_transport(u, t1, t2)
  assert np.allclose(u_transported, u), "Test case 2 failed"

  print("All test cases passed")


def crossMat(a):
    A=np.matrix([[0,- a[2],a[1]],[a[2],0,- a[0]],[- a[1],a[0],0]])
    return A