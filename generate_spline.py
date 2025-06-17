import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

import numpy as np
import splinepy as sp

nurbs = sp.NURBS(
    degrees=[3],
    knot_vectors=[[0, 0, 0, 0, 0.3, 0.6, 1, 1, 1, 1]],
    control_points=[
        [-1.0, 0.0, 0.0],
        [-0.5, 1.2, 0.0],
        [ 0.5, 1.0, 0.0],
        [ 1.2, 0.0, 0.0],
        [ 0.8,-0.8, 0.0],
        [ 0.0,-1.0, 0.0],
    ],
    weights=[1, 0.8, 0.8, 1, 1, 1],
)

queries = np.linspace(0, 1, 100).reshape(-1, 1)
points = nurbs.evaluate(queries)

plt.plot(points[:, 0], points[:, 1], label="NURBS curve")
plt.scatter(*zip(*[cp[:2] for cp in nurbs.control_points]), color='red', label="Control Points")
plt.legend()
plt.gca().set_aspect("equal")
plt.title("NURBS 2D curve (matplotlib)")
plt.show()
