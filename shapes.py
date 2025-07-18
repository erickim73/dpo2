# Letter shapes for spline transformation tasks
# Each template has 18 points that will be densified to match ctrl_state_dim // 2

# 1. Letter J (starting shape) → Letter E (target shape)
start_pts_template = np.array([
    [-0.1, 2.0],  # Top left of horizontal bar
    [0.6, 2.0],   # Top right of horizontal bar
    [0.6, 1.8],   # Top right corner
    [0.4, 1.8],   # Inner corner of bar
    [0.4, 1.6],   # Start of vertical descent
    [0.4, 1.3],   # Upper vertical
    [0.4, 1.0],   # Mid vertical
    [0.4, 0.7],   # Lower vertical
    [0.4, 0.4],   # Lower vertical before hook
    [0.4, 0.25],  # Hook transition
    [0.3, 0.1],   # Hook bend start
    [0.1, 0.0],   # Hook bend middle
    [-0.1, -0.05],# Bottom of hook
    [-0.3, 0.0],  # Left extent of hook
    [-0.4, 0.1],  # Hook left bottom curve
    [-0.4, 0.25], # Hook left side lower
    [-0.35, 0.35],# Hook left side upper
    [-0.25, 0.3]  # Hook closure
])

target_pts_template = np.array([
    [-0.35, -0.05], # Bottom left corner - rounded
    [-0.4, 0.1],    # Left bottom curve
    [-0.4, 0.6],    # Left side lower
    [-0.4, 1.2],    # Left side middle
    [-0.4, 1.8],    # Left side upper
    [-0.4, 2.0],    # Top left corner
    [-0.1, 2.0],    # Top left inner
    [0.2, 2.0],     # Top line middle
    [0.4, 2.0],     # Top right end
    [0.4, 1.7],     # Top right corner
    [0.1, 1.7],     # Top line return
    [-0.2, 1.7],    # Top line to left edge
    [-0.2, 1.3],    # Left edge upper
    [0.2, 1.3],     # Middle line end
    [0.3, 1.0],     # Middle line right edge
    [-0.2, 1.0],    # Middle to left edge
    [-0.2, 0.3],    # Left edge lower
    [0.35, -0.05]   # Bottom right corner - rounded
])

# 2. Letter C (starting shape) → Letter O (target shape)
start_pts_template = np.array([
    [0.5, 0.9], 
    [0.4, 1.5], 
    [0.0, 1.7], 
    [-0.4, 1.5],    
    [-0.5, 0.8],    
    [-0.5, 0.8],    
    [-0.4, 0.1],    
    [0.0, -0.1],
    [0.4, 0.1],     
    [0.5, 0.7],      
    [0.2, 0.8],     
    [0.1, 1.2],
    [-0.1, 1.3],    
    [-0.2, 0.9],    
    [-0.2, 0.7],    
    [-0.1, 0.3],    
    [0.1, 0.4],
    [0.2, 0.8],     
])

target_pts_template = np.array([
    [0.4, 0.8],
    [0.28, 1.28],   
    [0.0, 1.4],
    [-0.28, 1.28],  
    [-0.4, 0.8],
    [-0.4, 0.8],    
    [-0.28, 0.32],  
    [0.0, 0.2],
    [0.28, 0.32],   
    [0.4, 0.8],     
    [0.4, 0.8],     
    [0.15, 0.8],
    [0.11, 1.01],   
    [0.0, 1.05],
    [-0.11, 1.01],  
    [-0.15, 0.8],
    [-0.15, 0.8],   
    [-0.11, 0.59],  
    [0.0, 0.55],
    [0.11, 0.59],   
    [0.15, 0.8],    
])

# 3. Letter L (starting shape) → Letter T (target shape)
start_pts_template = np.array([
    # Outer boundary of L
    [-0.4, 1.5],    
    [-0.2, 1.5],    
    [-0.2, 0.2],    
    [0.4, 0.2],
    [0.4, 0.0],     
    [-0.4, 0.0],    
    [-0.4, 0.2],    
    [-0.4, 0.6],    
    [-0.4, 1.0],
    [-0.4, 1.5],    
    [-0.3, 1.3],
    [-0.3, 0.8],
    [-0.3, 0.1],    
    [0.3, 0.1],
    [0.2, 0.1],
    [-0.2, 0.1],    
    [-0.3, 0.4],    
    [-0.3, 1.3],    
])

target_pts_template = np.array([
    [-0.5, 1.5],
    [-0.2, 1.5],  
    [0.2, 1.5],   
    [0.5, 1.5],
    [0.5, 1.3],
    [0.3, 1.3],   
    [0.1, 1.3],
    [0.1, 0.65],  
    [0.1, 0.0],
    [0.0, 0.0],   
    [-0.1, 0.0],
    [-0.1, 0.65], 
    [-0.1, 1.3],
    [-0.3, 1.3],  
    [-0.5, 1.3],
    [-0.5, 1.4],   
    [-0.5, 1.5] 
])

# 4. Letter U (starting shape) → Letter D (target shape)
start_pts_template = np.array([
    [-0.5, 1.5],
    [-0.5, 0.5],
    [-0.4, 0.1],
    [-0.2, -0.1],
    [0.0, -0.1],
    [0.2, -0.1],
    [0.4, 0.1],
    [0.5, 0.5],
    [0.5, 1.5],
    [0.3, 1.5],
    [0.3, 0.6],
    [0.3, 0.2],
    [0.0, 0.1],
    [-0.3, 0.2],
    [-0.3, 0.6],
    [-0.3, 1.5],
    [-0.4, 1.5],
    [-0.5, 1.5],
])

target_pts_template = np.array([
    [-0.4, 1.5],
    [-0.4, 0.75],
    [-0.4, 0.0],
    [0.3, 0.0],
    [0.5, 0.75],
    [0.3, 1.5],
    [-0.2, 1.5],
    [-0.2, 1.3],
    [0.0, 1.3],
    [0.2, 1.1],
    [0.2, 0.4],
    [0.0, 0.2],
    [-0.2, 0.2],
    [-0.2, 0.4],
    [-0.2, 1.1],
    [-0.2, 1.3],
    [-0.2, 1.5],
    [-0.4, 1.5],
])

# 5. Letter P (starting shape) → Letter R (target shape)
start_pts_template = np.array([
    [-0.4, 2.0],    # Top left
    [-0.2, 2.0],    # Top left inner
    [0.1, 2.0],     # Top right inner
    [0.2, 1.8],     # Top right curve
    [0.2, 1.6],     # Right upper
    [0.2, 1.4],     # Right middle
    [0.1, 1.2],     # Right curve to middle
    [-0.2, 1.2],    # Middle horizontal
    [-0.2, 0.8],    # Vertical down
    [-0.2, 0.4],    # Vertical lower
    [-0.2, 0.0],    # Bottom inner
    [-0.4, 0.0],    # Bottom left
    [-0.4, 0.4],    # Left bottom
    [-0.4, 0.8],    # Left lower
    [-0.4, 1.2],    # Left middle
    [-0.4, 1.6],    # Left upper
    [-0.4, 2.0],    # Close at top
    [-0.3, 1.8],    # Inner top
])

target_pts_template = np.array([
    [-0.4, 2.0],    # Top left
    [-0.2, 2.0],    # Top left inner
    [0.1, 2.0],     # Top right inner
    [0.2, 1.8],     # Top right curve
    [0.2, 1.6],     # Right upper
    [0.2, 1.4],     # Right middle
    [0.1, 1.2],     # Right curve to middle
    [-0.2, 1.2],    # Middle horizontal
    [0.0, 1.0],     # Diagonal start
    [0.1, 0.8],     # Diagonal middle
    [0.2, 0.4],     # Diagonal lower
    [0.3, 0.0],     # Bottom right leg
    [0.1, 0.0],     # Leg inner
    [-0.1, 0.8],    # Back to vertical
    [-0.2, 0.4],    # Vertical lower
    [-0.2, 0.0],    # Bottom inner
    [-0.4, 0.0],    # Bottom left
    [-0.4, 1.8],    # Left side up
])

# Extended Letter shapes and other forms for spline transformation tasks
# Each template has 18 points that will be densified to match ctrl_state_dim // 2

# 7. Letter B (starting shape) → Letter P (target shape)
start_pts_template = np.array([
    [-0.4, 2.0],    # Top left
    [-0.2, 2.0],    # Top left inner
    [0.0, 2.0],     # Top center
    [0.2, 1.8],     # Top right curve
    [0.2, 1.6],     # Right upper
    [0.1, 1.4],     # Upper curve to middle
    [-0.2, 1.2],    # Middle horizontal
    [0.0, 1.2],     # Middle center
    [0.2, 1.0],     # Middle right curve
    [0.3, 0.8],     # Right lower
    [0.3, 0.4],     # Right bottom
    [0.2, 0.2],     # Bottom right curve
    [0.0, 0.0],     # Bottom center
    [-0.2, 0.0],    # Bottom left inner
    [-0.4, 0.0],    # Bottom left
    [-0.4, 0.8],    # Left lower
    [-0.4, 1.4],    # Left middle
    [-0.4, 1.8],    # Left upper
])

target_pts_template = np.array([
    [-0.4, 2.0],    # Top left
    [-0.2, 2.0],    # Top left inner
    [0.1, 2.0],     # Top right inner
    [0.2, 1.8],     # Top right curve
    [0.2, 1.6],     # Right upper
    [0.2, 1.4],     # Right middle
    [0.1, 1.2],     # Right curve to middle
    [-0.2, 1.2],    # Middle horizontal
    [-0.2, 0.8],    # Vertical down
    [-0.2, 0.4],    # Vertical lower
    [-0.2, 0.0],    # Bottom inner
    [-0.4, 0.0],    # Bottom left
    [-0.4, 0.4],    # Left bottom
    [-0.4, 0.8],    # Left lower
    [-0.4, 1.2],    # Left middle
    [-0.4, 1.6],    # Left upper
    [-0.4, 2.0],    # Close at top
    [-0.3, 1.8],    # Inner top
])

# 8. Letter A (starting shape) → Letter V (target shape)
start_pts_template = np.array([
    [0.0, 2.0],     # Top peak
    [-0.1, 1.8],    # Left upper
    [-0.2, 1.6],    # Left upper middle
    [-0.3, 1.4],    # Left middle
    [-0.4, 1.2],    # Left lower middle
    [-0.4, 1.0],    # Left crossbar level
    [-0.2, 1.0],    # Crossbar left
    [0.2, 1.0],     # Crossbar right
    [0.4, 1.0],     # Right crossbar level
    [0.4, 1.2],     # Right lower middle
    [0.3, 1.4],     # Right middle
    [0.2, 1.6],     # Right upper middle
    [0.1, 1.8],     # Right upper
    [0.0, 2.0],     # Back to peak
    [-0.5, 0.0],    # Bottom left
    [-0.3, 0.0],    # Left foot inner
    [0.3, 0.0],     # Right foot inner
    [0.5, 0.0],     # Bottom right
])

target_pts_template = np.array([
    [0.0, 2.0],     # Top peak
    [-0.1, 1.8],    # Left upper
    [-0.2, 1.6],    # Left upper middle
    [-0.3, 1.4],    # Left middle
    [-0.4, 1.2],    # Left lower middle
    [-0.45, 1.0],   # Left lower
    [-0.5, 0.8],    # Left bottom area
    [-0.5, 0.4],    # Left bottom
    [-0.5, 0.0],    # Bottom left
    [-0.3, 0.0],    # Left inner
    [-0.1, 0.6],    # Left inner rise
    [0.0, 1.2],     # Center point
    [0.1, 0.6],     # Right inner rise
    [0.3, 0.0],     # Right inner
    [0.5, 0.0],     # Bottom right
    [0.5, 0.4],     # Right bottom
    [0.5, 0.8],     # Right bottom area
    [0.45, 1.0],    # Right lower
])

# 9. Circle (starting shape) → Square (target shape)
start_pts_template = np.array([
    [0.0, 1.0],     # Top
    [0.2, 0.98],    # Top right
    [0.38, 0.92],   # Right upper
    [0.52, 0.82],   # Right upper middle
    [0.64, 0.68],   # Right middle
    [0.74, 0.52],   # Right lower middle
    [0.82, 0.34],   # Right lower
    [0.88, 0.16],   # Right bottom
    [0.9, 0.0],     # Right
    [0.88, -0.16],  # Right bottom neg
    [0.82, -0.34],  # Right lower neg
    [0.74, -0.52],  # Right lower middle neg
    [0.64, -0.68],  # Right middle neg
    [0.52, -0.82],  # Right upper middle neg
    [0.38, -0.92],  # Right upper neg
    [0.2, -0.98],   # Top right neg
    [0.0, -1.0],    # Bottom
    [-0.2, -0.98],  # Bottom left
])

target_pts_template = np.array([
    [0.0, 1.0],     # Top center
    [0.3, 1.0],     # Top right
    [0.6, 1.0],     # Top right corner
    [0.9, 1.0],     # Top right edge
    [0.9, 0.7],     # Right top
    [0.9, 0.4],     # Right upper middle
    [0.9, 0.1],     # Right middle
    [0.9, -0.2],    # Right lower middle
    [0.9, -0.5],    # Right lower
    [0.9, -0.8],    # Right bottom
    [0.9, -1.0],    # Right bottom corner
    [0.6, -1.0],    # Bottom right
    [0.3, -1.0],    # Bottom right middle
    [0.0, -1.0],    # Bottom center
    [-0.3, -1.0],   # Bottom left middle
    [-0.6, -1.0],   # Bottom left
    [-0.9, -1.0],   # Bottom left corner
    [-0.9, -0.7],   # Left bottom
])

# 10. Triangle (starting shape) → Diamond (target shape)
start_pts_template = np.array([
    [0.0, 1.0],     # Top peak
    [-0.1, 0.9],    # Left upper
    [-0.2, 0.8],    # Left upper middle
    [-0.3, 0.7],    # Left middle
    [-0.4, 0.6],    # Left lower middle
    [-0.5, 0.5],    # Left lower
    [-0.6, 0.4],    # Left bottom area
    [-0.7, 0.3],    # Left bottom
    [-0.8, 0.2],    # Left bottom edge
    [-0.9, 0.0],    # Bottom left
    [-0.45, 0.0],   # Bottom left middle
    [0.0, 0.0],     # Bottom center
    [0.45, 0.0],    # Bottom right middle
    [0.9, 0.0],     # Bottom right
    [0.8, 0.2],     # Right bottom edge
    [0.7, 0.3],     # Right bottom
    [0.6, 0.4],     # Right bottom area
    [0.5, 0.5],     # Right lower
])

target_pts_template = np.array([
    [0.0, 1.0],     # Top peak
    [-0.1, 0.9],    # Top left
    [-0.2, 0.8],    # Left upper
    [-0.3, 0.7],    # Left upper middle
    [-0.4, 0.6],    # Left middle
    [-0.5, 0.5],    # Left center
    [-0.6, 0.4],    # Left lower middle
    [-0.7, 0.3],    # Left lower
    [-0.8, 0.2],    # Left bottom
    [-0.9, 0.0],    # Left peak
    [-0.8, -0.2],   # Left bottom neg
    [-0.7, -0.3],   # Left lower neg
    [-0.6, -0.4],   # Left lower middle neg
    [-0.5, -0.5],   # Left center neg
    [-0.4, -0.6],   # Left middle neg
    [-0.3, -0.7],   # Left upper middle neg
    [-0.2, -0.8],   # Left upper neg
    [-0.1, -0.9],   # Bottom left
])

# 11. Star (starting shape) → Plus/Cross (target shape)
start_pts_template = np.array([
    [0.0, 1.0],     # Top point
    [0.15, 0.7],    # Top right inner
    [0.5, 0.8],     # Right upper point
    [0.2, 0.5],     # Right inner
    [0.6, 0.3],     # Right point
    [0.25, 0.15],   # Right lower inner
    [0.4, -0.2],    # Bottom right point
    [0.1, 0.0],     # Bottom right inner
    [0.0, -0.5],    # Bottom point
    [-0.1, 0.0],    # Bottom left inner
    [-0.4, -0.2],   # Bottom left point
    [-0.25, 0.15],  # Left lower inner
    [-0.6, 0.3],    # Left point
    [-0.2, 0.5],    # Left inner
    [-0.5, 0.8],    # Left upper point
    [-0.15, 0.7],   # Top left inner
    [0.0, 1.0],     # Back to top
    [0.0, 0.5],     # Center
])

target_pts_template = np.array([
    [0.0, 1.0],     # Top
    [-0.1, 1.0],    # Top left
    [-0.2, 1.0],    # Top left edge
    [-0.2, 0.8],    # Top left inner
    [-0.2, 0.6],    # Left upper
    [-0.2, 0.4],    # Left upper middle
    [-0.2, 0.2],    # Left center
    [-0.4, 0.2],    # Left horizontal
    [-0.6, 0.2],    # Left outer
    [-0.8, 0.2],    # Left edge
    [-0.8, 0.0],    # Left center line
    [-0.8, -0.2],   # Left edge neg
    [-0.6, -0.2],   # Left outer neg
    [-0.4, -0.2],   # Left horizontal neg
    [-0.2, -0.2],   # Left center neg
    [-0.2, -0.4],   # Left lower middle
    [-0.2, -0.6],   # Left lower
    [-0.2, -0.8],   # Left bottom inner
])

# 12. Hexagon (starting shape) → Oval (target shape)
start_pts_template = np.array([
    [0.0, 1.0],     # Top
    [0.43, 0.75],   # Top right
    [0.75, 0.43],   # Right upper
    [0.87, 0.0],    # Right
    [0.75, -0.43],  # Right lower
    [0.43, -0.75],  # Bottom right
    [0.0, -1.0],    # Bottom
    [-0.43, -0.75], # Bottom left
    [-0.75, -0.43], # Left lower
    [-0.87, 0.0],   # Left
    [-0.75, 0.43],  # Left upper
    [-0.43, 0.75],  # Top left
    [0.0, 1.0],     # Back to top
    [0.22, 0.87],   # Inner top right
    [0.65, 0.22],   # Inner right
    [0.22, -0.87],  # Inner bottom right
    [-0.22, -0.87], # Inner bottom left
    [-0.65, -0.22], # Inner left
])

target_pts_template = np.array([
    [0.0, 1.0],     # Top
    [0.2, 0.98],    # Top right
    [0.38, 0.92],   # Right upper
    [0.52, 0.82],   # Right upper middle
    [0.64, 0.68],   # Right middle
    [0.74, 0.52],   # Right lower middle
    [0.82, 0.34],   # Right lower
    [0.88, 0.16],   # Right bottom
    [0.9, 0.0],     # Right
    [0.88, -0.16],  # Right bottom neg
    [0.82, -0.34],  # Right lower neg
    [0.74, -0.52],  # Right lower middle neg
    [0.64, -0.68],  # Right middle neg
    [0.52, -0.82],  # Right upper middle neg
    [0.38, -0.92],  # Right upper neg
    [0.2, -0.98],   # Top right neg
    [0.0, -1.0],    # Bottom
    [-0.2, -0.98],  # Bottom left
])

# 13. Spiral (starting shape) → Concentric Circles (target shape)
start_pts_template = np.array([
    [0.0, 0.0],     # Center start
    [0.1, 0.05],    # Spiral out
    [0.15, 0.15],   # Spiral curve
    [0.05, 0.25],   # Spiral loop
    [-0.1, 0.2],    # Spiral back
    [-0.2, 0.0],    # Spiral left
    [-0.15, -0.25], # Spiral down
    [0.1, -0.3],    # Spiral right
    [0.35, -0.1],   # Spiral out right
    [0.4, 0.2],     # Spiral up
    [0.2, 0.45],    # Spiral top
    [-0.15, 0.5],   # Spiral left top
    [-0.5, 0.3],    # Spiral far left
    [-0.6, -0.1],   # Spiral bottom left
    [-0.4, -0.5],   # Spiral bottom
    [0.2, -0.6],    # Spiral bottom right
    [0.7, -0.2],    # Spiral far right
    [0.8, 0.4],     # Spiral end
])

target_pts_template = np.array([
    [0.0, 0.3],     # Inner circle top
    [0.15, 0.26],   # Inner circle right
    [0.21, 0.15],   # Inner circle bottom right
    [0.21, 0.0],    # Inner circle right
    [0.15, -0.15],  # Inner circle bottom
    [0.0, -0.21],   # Inner circle bottom
    [-0.15, -0.15], # Inner circle bottom left
    [-0.21, 0.0],   # Inner circle left
    [-0.15, 0.15],  # Inner circle top left
    [0.0, 0.3],     # Back to inner top
    [0.0, 0.8],     # Outer circle top
    [0.4, 0.7],     # Outer circle right
    [0.57, 0.4],    # Outer circle bottom right
    [0.57, 0.0],    # Outer circle right
    [0.4, -0.4],    # Outer circle bottom
    [0.0, -0.57],   # Outer circle bottom
    [-0.4, -0.4],   # Outer circle bottom left
    [-0.57, 0.0],   # Outer circle left
])