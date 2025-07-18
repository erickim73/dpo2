import numpy as np
import math

def ccw(a,b,c):
    return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

def segments_intersect(p1,p2,p3,p4):
    # return True if segment p1–p2 crosses p3–p4
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def _segment_distance(self, p1, p2, p3, p4):
    """
    Calculate minimum distance between two line segments
    """
    # Convert to numpy arrays for easier math
    p1, p2, p3, p4 = map(np.array, [p1, p2, p3, p4])
    
    # Vector representations
    v1 = p2 - p1  # Direction of segment 1
    v2 = p4 - p3  # Direction of segment 2
    w = p1 - p3   # Vector between segment starts
    
    a = np.dot(v1, v1)  # |v1|^2
    b = np.dot(v1, v2)  # v1 · v2
    c = np.dot(v2, v2)  # |v2|^2
    d = np.dot(v1, w)   # v1 · w
    e = np.dot(v2, w)   # v2 · w
    
    denom = a * c - b * b
    
    if abs(denom) < 1e-10:  # Parallel segments
        # Distance between parallel lines
        return np.linalg.norm(np.cross(w, v1)) / np.linalg.norm(v1)
    
    # Parameters for closest points
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    
    # Clamp to segment bounds
    s = np.clip(s, 0, 1)
    t = np.clip(t, 0, 1)
    
    # Closest points
    closest1 = p1 + s * v1
    closest2 = p3 + t * v2
    
    return np.linalg.norm(closest1 - closest2)