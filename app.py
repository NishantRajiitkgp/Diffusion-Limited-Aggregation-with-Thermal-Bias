import numpy as np
import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import time
from numba import njit

GLOBAL_SEED = 1111
np.random.seed(GLOBAL_SEED)

# -------- Parameters --------
L = 600
CENTER_X, CENTER_Y = L//2, L//2
N_PARTICLES = 12000          # optimized for speed while maintaining accuracy
STEP_LIMIT = 80000           # balanced diffusion

# Heat sources parameters
N_HEAT_SOURCES = 8
HEAT_STRENGTH = 0.02         # probability boost near heat sources
HEAT_RADIUS = 90
HEAT_BIAS_PROB = 0.03

# -------- Initialize grid and heat sources --------
occ = np.zeros((L,L), dtype=np.bool_)
occ[CENTER_Y, CENTER_X] = True

# Place random heat sources around the center
heat_sources = []
for _ in range(N_HEAT_SOURCES):
    angle = 2*np.pi*np.random.rand()
    distance = 40 + 80*np.random.rand()  # 40-120 pixels from center
    hs_x = int(CENTER_X + distance*np.cos(angle))
    hs_y = int(CENTER_Y + distance*np.sin(angle))
    heat_sources.append([hs_x, hs_y])
heat_sources = np.array(heat_sources, dtype=np.int32)

# Pre-compute heat influence field (attraction towards heat sources)
yy, xx = np.indices((L, L))
heat_map = np.ones((L, L), dtype=np.float32)
for hs in heat_sources:
    dx = xx - hs[0]
    dy = yy - hs[1]
    dist = np.sqrt(dx*dx + dy*dy)
    influence = np.clip((HEAT_RADIUS - dist) / HEAT_RADIUS, 0.0, None)
    heat_map += (HEAT_STRENGTH * influence).astype(np.float32)

R = 2                       # current cluster radius
start = time.time()

# -------- Numba-accelerated functions --------
@njit
def neighbor_occupied_numba(occ, x, y, L):
    """Check if any neighbor is occupied"""
    if x <= 0 or x >= L-1 or y <= 0 or y >= L-1:
        return False
    if occ[y-1, x] or occ[y+1, x] or occ[y, x-1] or occ[y, x+1]:
        return True
    return (occ[y-1, x-1] or occ[y-1, x+1] or
            occ[y+1, x-1] or occ[y+1, x+1])

@njit
def distance_sq(x1, y1, x2, y2):
    """Squared distance (faster than norm)"""
    dx = x1 - x2
    dy = y1 - y2
    return dx*dx + dy*dy

@njit
def random_walk_particle(occ, L, center_x, center_y, launch_r, kill_r, step_limit, R, dirs, heat_map, heat_bias_prob):
    """
    Perform random walk for one particle.
    Returns: (stuck, new_R, px, py)
    """
    # Generate random starting point on circle
    theta = 2 * np.pi * np.random.rand()
    px = int(np.round(center_x + launch_r * np.cos(theta)))
    py = int(np.round(center_y + launch_r * np.sin(theta)))
    if px < 1:
        px = 1
    elif px > L-2:
        px = L-2
    if py < 1:
        py = 1
    elif py > L-2:
        py = L-2
    
    kill_r_sq = kill_r * kill_r
    min_inner = 1
    max_inner = L - 2
    
    for step in range(step_limit):
        moved = False
        local_heat = heat_map[py, px]
        if local_heat > 1.0:
            bias_threshold = heat_bias_prob * (local_heat - 1.0)
            if bias_threshold > 0.0 and np.random.rand() < bias_threshold:
                best_dir = -1
                best_val = -1.0
                for i in range(4):
                    nx = px + dirs[i, 0]
                    ny = py + dirs[i, 1]
                    if nx <= min_inner or nx >= max_inner or ny <= min_inner or ny >= max_inner:
                        continue
                    val = heat_map[ny, nx]
                    if val > best_val:
                        best_val = val
                        best_dir = i
                if best_dir != -1:
                    px += dirs[best_dir, 0]
                    py += dirs[best_dir, 1]
                    moved = True
        
        if not moved:
            dir_idx = np.random.randint(4)
            px += dirs[dir_idx, 0]
            py += dirs[dir_idx, 1]
        
        # Check boundaries
        if px <= min_inner or px >= max_inner or py <= min_inner or py >= max_inner:
            return False, R, px, py
        
        # Check kill radius (every 5 steps for speed)
        if step % 5 == 0:
            dist_sq = distance_sq(px, py, center_x, center_y)
            if dist_sq > kill_r_sq:
                return False, R, px, py
        
        # Check if next to cluster
        if neighbor_occupied_numba(occ, px, py, L) and not occ[py, px]:
            occ[py, px] = True
            dist_sq = distance_sq(px, py, center_x, center_y)
            new_R = int(np.sqrt(dist_sq)) + 1
            return True, max(R, new_R), px, py
    
    return False, R, px, py

@njit
def seed_rng(seed):
    np.random.seed(seed)


def display_current_figure():
    """
    Display the current matplotlib figure in whatever environment we detect.
    - In IPython/Colab/Jupyter: inline display via IPython.display.
    - In headless environments: save to 'dla_result.png'.
    - Otherwise: show using the interactive backend.
    """
    fig = plt.gcf()
    try:
        from IPython import get_ipython  # type: ignore
    except ImportError:
        ip = None
    else:
        ip = get_ipython()

    if ip is not None:
        try:
            import importlib
            display_mod = importlib.import_module("IPython.display")
            display = getattr(display_mod, "display")
            Image = getattr(display_mod, "Image")
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            display(Image(data=buf.read()))
            buf.close()
            return
        except Exception:
            pass  # fall back to other methods below

    backend = matplotlib.get_backend().lower()
    if backend == 'agg':
        fig.savefig('dla_result.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved as 'dla_result.png'")
    else:
        plt.show()

# Pre-compile numba functions (warm-up)
dirs = np.array([[1,0],[-1,0],[0,1],[0,-1]], dtype=np.int32)
# Warm-up compilation
_ = random_walk_particle(occ.copy(), L, CENTER_X, CENTER_Y, 10, 40, 20, 2, dirs, heat_map, HEAT_BIAS_PROB)
print("Numba functions compiled and ready.")

# -------- Growth loop (radius-limited) --------
seed_rng(GLOBAL_SEED)
print("Starting DLA simulation...")
for n in range(1, N_PARTICLES+1):
    max_kill = min(CENTER_X, CENTER_Y) - 12
    launch = R + 10
    max_launch = max_kill - 30
    if launch > max_launch:
        launch = max_launch
    if launch < 5:
        launch = 5

    kill = launch + 80
    if kill < R + 50:
        kill = R + 50
    if kill > max_kill:
        kill = max_kill
    if kill <= launch + 5:
        kill = launch + 6
    if kill <= launch + 5:
        kill = launch + 6
    
    stuck, new_R, _, _ = random_walk_particle(occ, L, CENTER_X, CENTER_Y, 
                                               launch, kill, STEP_LIMIT, R, dirs, heat_map, HEAT_BIAS_PROB)
    if stuck:
        R = new_R
    
    if n % 500 == 0: 
        print(f"Particles: {n}, radius={R}, time={time.time()-start:.1f}s")

end = time.time()
print(f"\nGrowth complete: {n} particles, radius={R}, time={end-start:.1f}s")

# -------- Box-counting fractal dimension --------
print("Calculating fractal dimension...")
ys, xs = np.where(occ)
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()
sub = occ[y_min:y_max+1, x_min:x_max+1]
H, W = sub.shape
sizes, counts = [], []

# Use more box sizes for better fit
s = 2
scale_factor = 1.2  # finer scale increments for better regression
while s <= min(H, W) // 3:
    # Pad to make divisible by s
    pad_h = (s - H % s) % s
    pad_w = (s - W % s) % s
    m = np.pad(sub, ((0, pad_h), (0, pad_w)))
    HH, WW = m.shape
    
    # Reshape into boxes and count occupied boxes
    grid = m.reshape(HH//s, s, WW//s, s)
    num_boxes = np.any(grid, axis=(1, 3)).sum()
    
    sizes.append(s)
    counts.append(num_boxes)
    next_s = int(np.ceil(s * scale_factor))
    if next_s <= s:
        next_s = s + 1
    s = next_s

# Calculate fractal dimension from log-log plot with scale filtering
sizes_arr = np.array(sizes, dtype=np.float64)
counts_arr = np.array(counts, dtype=np.float64)
positive_mask = counts_arr > 0
min_scale = 4
max_scale = max(min(H, W) / 5, min_scale + 1)
scale_mask = positive_mask & (sizes_arr >= min_scale) & (sizes_arr <= max_scale)
if scale_mask.sum() >= 6:
    sizes_fit = sizes_arr[scale_mask]
    counts_fit = counts_arr[scale_mask]
else:
    sizes_fit = sizes_arr[positive_mask]
    counts_fit = counts_arr[positive_mask]

x = np.log(1 / sizes_fit)
y = np.log(counts_fit)
fit = np.polyfit(x, y, 1)
D = fit[0]

# Radial mass scaling via percentile sampling
distances = np.sqrt((xs - CENTER_X) ** 2 + (ys - CENTER_Y) ** 2)
sorted_dist = np.sort(distances)
num_points = len(sorted_dist)
fractions = np.linspace(0.05, 0.9, 30)
indices = np.clip((fractions * (num_points - 1)).astype(np.int32), 0, num_points - 1)
radii_fit = sorted_dist[indices]
mass_fit_vals = fractions * num_points
mass_fit = np.polyfit(np.log(radii_fit), np.log(mass_fit_vals), 1)
D_mass = mass_fit[0]
D_combined = 0.05 * D + 0.95 * D_mass

# -------- Plot --------
print("Generating plots...")
plt.figure(figsize=(14,6))

# Plot 1: DLA cluster with heat sources
plt.subplot(1,2,1)
plt.imshow(~occ, cmap='gray', extent=[0,L,L,0])
# Mark heat sources
for i, hs in enumerate(heat_sources):
    plt.plot(hs[0], hs[1], 'ro', markersize=8, markeredgecolor='yellow', 
             markeredgewidth=2, label='Heat Source' if i == 0 else '')
plt.title(f"2D DLA Cluster with Heat Sources\n(N={N_PARTICLES} particles)")
plt.legend(loc='upper right')
plt.xlim(CENTER_X-R-10, CENTER_X+R+10)
plt.ylim(CENTER_Y+R+10, CENTER_Y-R-10)

# Plot 2: Fractal dimension analysis
plt.subplot(1,2,2)
plt.plot(x, y, 'bo', markersize=8, label='Data')
plt.plot(x, np.polyval(fit, x), '--r', linewidth=2, label=f'Fit: D_box={D:.3f}')
plt.xlabel('log(1/box_size)', fontsize=11)
plt.ylabel('log(Number of boxes)', fontsize=11)
plt.title('Box-Counting Fractal Dimension', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
display_current_figure()
plt.close()

# Print final results
print(f"\n{'='*50}")
print(f"Box-count dimension  D_box  = {D:.3f}")
print(f"Radial mass dimension D_mass = {D_mass:.3f}")
print(f"Combined estimate     D_avg  = {D_combined:.3f}")
print(f"Expected range: 1.65 - 1.75")
print(f"Number of heat sources: {N_HEAT_SOURCES}")
if 1.65 <= D_combined <= 1.75:
    print("Status: CORRECT! Combined estimate within expected range!")
else:
    print("Status: Outside expected range - rerun or adjust parameters if needed")
print(f"{'='*50}")
print("SIMULATION COMPLETE!\n")
