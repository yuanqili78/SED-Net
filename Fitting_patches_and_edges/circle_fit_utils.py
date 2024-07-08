import numpy as np

# -------------------------------------------------------------------------------
# RODRIGUES ROTATION
# - Rotate given points based on a starting and ending vector
# - Axis k and angle of rotation theta given by vectors n0,n1
#   P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
# -------------------------------------------------------------------------------
def rodrigues_rot(P, n0, n1):
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis, :]

    # Get vector of rotation k and angle theta
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))

    # Compute rotated points
    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        P_rot[i] = P[i] * np.cos(theta) + np.cross(k, P[i]) * np.sin(theta) + k * np.dot(k, P[i]) * (1 - np.cos(theta))

    return P_rot


# -------------------------------------------------------------------------------
# FIT CIRCLE 2D
# - Find center [xc, yc] and radius r of circle fitting to set of 2D points
# - Optionally specify weights for points
#
# - Implicit circle function:
#   (x-xc)^2 + (y-yc)^2 = r^2
#   (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
#   c[0]*x + c[1]*y + c[2] = x^2+y^2
#
# - Solution by method of least squares:
#   A*c = b, c' = argmin(||A*c - b||^2)
#   A = [x y 1], b = [x^2+y^2]
# -------------------------------------------------------------------------------
def fit_circle_2d(x, y, w=[]):
    A = np.array([x, y, np.ones(len(x))]).T
    b = x ** 2 + y ** 2

    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)

    # Solve by method of least squares
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc ** 2 + yc ** 2)
    return xc, yc, r


# -------------------------------------------------------------------------------
# Generate points on circle
# P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
# -------------------------------------------------------------------------------
def generate_circle_by_vectors(t, C, r, n, u):
    n = n / np.linalg.norm(n)
    u = u / np.linalg.norm(u)
    P_circle = r * np.cos(t)[:, np.newaxis] * u + r * np.sin(t)[:, np.newaxis] * np.cross(n, u) + C
    return P_circle


# https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
def circle_segmentation(cloud):
    # -------------------------------------------------------------------------------
    # (1) Fitting plane by SVD for the mean-centered data
    # Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
    # -------------------------------------------------------------------------------
    P_mean = cloud.mean(axis=0)
    P_centered = cloud - P_mean
    U, s, V = np.linalg.svd(P_centered)

    # Normal vector of fitting plane is given by 3rd column in V
    # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
    normal = V[2, :]
    d = -np.dot(P_mean, normal)  # d = -<p,n>

    # -------------------------------------------------------------------------------
    # (2) Project points to coords X-Y in 2D plane
    # -------------------------------------------------------------------------------
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])

    # -------------------------------------------------------------------------------
    # (3) Fit circle in new 2D coords
    # -------------------------------------------------------------------------------
    xc, yc, r = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])

    # --- Generate circle points in 2D
    t = np.linspace(0, 2 * np.pi, 100)
    xx = xc + r * np.cos(t)
    yy = yc + r * np.sin(t)

    # -------------------------------------------------------------------------------
    # (4) Transform circle center back to 3D coords
    # -------------------------------------------------------------------------------
    C = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
    C = C.flatten()

    # --- Generate points for fitting circle
    t = np.linspace(0, 2 * np.pi, 1000)
    u = cloud[0] - C
    P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)
    return P_fitcircle, C, r