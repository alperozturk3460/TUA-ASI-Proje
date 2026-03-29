import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import scipy.ndimage as ndimage
import time

# --- FİZİKSEL VE STRATEJİK SABİTLER ---
DEM_SIZE = 100               
PIXEL_TO_METER = 0.5         
SAFE_SLOPE = 14.0            
DANGER_SLOPE = 38.0          
DISTANCE_WEIGHT = 1.1        
RISK_FACTOR = 6.0            
BATTERY_CAPACITY = 400.0     
SUN_VECTOR = np.array([1, 1, 0.5])
SUN_VECTOR = SUN_VECTOR / np.linalg.norm(SUN_VECTOR)

# --- GLOBAL DEĞİŞKENLER ---
height_map, true_hazards, known_hazards, seen_mask, raw_img = None, None, None, None, None
reset_requested = False
science_samples = []
collected_samples = 0
mission_start_time = 0
mission_log = ["Sistem Başlatıldı...", "Harita Yüklendi."]

def generate_lunar_terrain(size):
    noise_base = np.random.rand(size, size)
    terrain = ndimage.gaussian_filter(noise_base, sigma=10.0) * 12.0
    y_idx, x_idx = np.indices((size, size))
    for _ in range(10): 
        kx, ky = np.random.randint(10, size-10, 2)
        r = np.random.randint(8, 16)
        depth = np.random.uniform(4.0, 7.5)
        dist_sq = (x_idx - kx)**2 + (y_idx - ky)**2
        mask_bowl = dist_sq < (r * 0.85)**2
        mask_rim = (dist_sq >= (r * 0.85)**2) & (dist_sq < (r * 1.4)**2)
        dist_bowl = np.sqrt(dist_sq[mask_bowl])
        terrain[mask_bowl] -= (depth * (1 - (dist_bowl / (r * 0.85))**2))
        dist_rim = np.sqrt(dist_sq[mask_rim])
        terrain[mask_rim] += (depth * 0.35 * (1 - (abs(dist_rim - r*1.1) / (r * 0.3))**2))
    return (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 15.0

def calculate_slope_map(dem, scale):
    dy, dx = np.gradient(dem, scale)
    return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

def create_shaded_relief(dem, sun_vec):
    dy, dx = np.gradient(dem)
    normals = np.stack([-dx, -dy, np.ones_like(dem)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    intensity = np.tensordot(normals, sun_vec, axes=(2, 0))
    return (intensity - intensity.min()) / (intensity.max() - intensity.min())

def add_log(message):
    global mission_log
    mission_log.append(message)
    if len(mission_log) > 6:
        mission_log.pop(0)

def generate_mission():
    global height_map, true_hazards, known_hazards, seen_mask, raw_img, science_samples, collected_samples, mission_start_time, mission_log
    height_map = generate_lunar_terrain(DEM_SIZE)
    slope_map = calculate_slope_map(height_map, PIXEL_TO_METER)
    true_hazards = np.where(slope_map > DANGER_SLOPE, 1, 0)
    known_hazards = np.zeros((DEM_SIZE, DEM_SIZE), dtype=np.int8)
    seen_mask = np.zeros((DEM_SIZE, DEM_SIZE), dtype=np.int8)
    raw_img = create_shaded_relief(height_map, SUN_VECTOR)
    
    science_samples = []
    attempts = 0
    while len(science_samples) < 3 and attempts < 200:
        y, x = np.random.randint(15, 85, 2)
        if slope_map[y, x] < SAFE_SLOPE * 0.8 and true_hazards[y, x] == 0:
            science_samples.append([y, x, False])
        attempts += 1
    while len(science_samples) < 3:
        science_samples.append([np.random.randint(15, 85), np.random.randint(15, 85), False])
        
    collected_samples = 0
    mission_start_time = time.time()
    mission_log = ["Sistem Başlatıldı...", "Hedefler Tanımlandı."]

def a_star_fast(known_grid, dem, start, goal):
    rows, cols = known_grid.shape
    dist_map = ndimage.distance_transform_edt(known_grid == 0) if np.any(known_grid) else np.ones((rows, cols)) * 50
    open_list = [(0, start)]; came_from, g_score = {}, {start: 0}
    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            path = []
            while current in came_from: path.append(current); current = came_from[current]
            return path[::-1]
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                d_move = 1.414 if dx != 0 and dy != 0 else 1.0
                move_cost = d_move * DISTANCE_WEIGHT
                h_diff = abs(dem[neighbor] - dem[current])
                slope = np.degrees(np.arctan(h_diff / (d_move * PIXEL_TO_METER)))
                if slope > DANGER_SLOPE: continue
                slope_penalty = ((slope - SAFE_SLOPE) ** 1.8) * RISK_FACTOR if slope > SAFE_SLOPE else 0
                prox_penalty = (2.5 - dist_map[neighbor]) * 50 if dist_map[neighbor] < 2.5 else 0
                new_g = g_score[current] + move_cost + slope_penalty + prox_penalty
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    came_from[neighbor], g_score[neighbor] = current, new_g
                    h = np.sqrt((neighbor[0]-goal[0])**2 + (neighbor[1]-goal[1])**2) * 1.1
                    heapq.heappush(open_list, (new_g + h, neighbor))
    return None

def change_map_callback(event):
    global reset_requested
    reset_requested = True

# --- ANA KURULUM ---
plt.ion()
fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[5, 1])
ax_map = fig.add_subplot(gs[:, 0])
ax_telemetry = fig.add_subplot(gs[0, 1])
ax_telemetry.set_axis_off()

ax_button = plt.axes([0.76, 0.05, 0.20, 0.07])
btn_change = Button(ax_button, 'YENİ GÖREV / RESET', color='#1a252f', hovercolor='#2c3e50')
btn_change.label.set_color('white')
btn_change.on_clicked(change_map_callback)

while True:
    generate_mission()
    reset_requested, current_pos = False, (5, 5)
    final_goal = (DEM_SIZE-8, DEM_SIZE-8)
    history, battery, total_dist, step = [current_pos], 400.0, 0.0, 0
    current_path = []

    ax_map.clear()
    ax_map.set_title("ARTEMIS MISSION CONTROL - LUNAR ROVER NAV", fontweight='bold', pad=15)
    img_obj = ax_map.imshow(raw_img, cmap='gray', origin='upper')
    hazard_obj = ax_map.imshow(np.zeros((DEM_SIZE, DEM_SIZE, 4)), origin='upper')
    path_line, = ax_map.plot([], [], color='#00FF00', linewidth=1.2, alpha=0.6)
    plan_line, = ax_map.plot([], [], color='cyan', linestyle=':', alpha=0.4)
    rover_marker = ax_map.scatter([current_pos[1]], [current_pos[0]], color='white', s=150, edgecolors='black', zorder=10)
    samples_marker = ax_map.scatter([], [], s=100, edgecolors='black', zorder=5)
    goal_marker = ax_map.scatter([final_goal[1]], [final_goal[0]], color='#f1c40f', marker='*', s=250, zorder=5)
    
    telemetry_text = ax_telemetry.text(0.05, 0.5, "", fontsize=10, family='monospace', color='#00FF00', 
                                     bbox=dict(facecolor='black', boxstyle='round,pad=1', alpha=0.8), va='center')

    while not reset_requested and battery > 0:
        active_samples = [s[:2] for s in science_samples if not s[2]]
        target = final_goal
        status = "ANA HEDEF"
        if active_samples:
            dists = [np.linalg.norm(np.array(current_pos)-np.array(s)) for s in active_samples]
            target = tuple(active_samples[np.argmin(dists)])
            status = "ÖRNEK TOPLAMA"

        y_g, x_g = np.ogrid[:DEM_SIZE, :DEM_SIZE]
        l_mask = (y_g - current_pos[0])**2 + (x_g - current_pos[1])**2 <= 18**2
        seen_mask[l_mask] = 1
        
        new_hazards_mask = (true_hazards[l_mask] == 1) & (known_hazards[l_mask] == 0)
        replan_needed = False
        if np.any(new_hazards_mask):
            known_hazards[l_mask] = true_hazards[l_mask]
            add_log("Yeni Engel Tespit Edildi!")
            if current_path:
                for pt in current_path[:10]:
                    if known_hazards[int(pt[0]), int(pt[1])] == 1:
                        replan_needed = True; break
            else: replan_needed = True

        if replan_needed or not current_path:
            current_path = a_star_fast(known_hazards, height_map, current_pos, target)
        
        if current_path:
            prev_pos, current_pos = current_pos, current_path.pop(0)
            h_diff = height_map[current_pos] - height_map[prev_pos]
            d_p = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
            slope = np.degrees(np.arctan(abs(h_diff)/(d_p * PIXEL_TO_METER))) if d_p > 0 else 0
            
            # --- GELİŞMİŞ GÜÇ MODELİ ---
            solar_gain = 0.20 * raw_img[current_pos] # Güneşten gelen güç (0-0.2W)
            base_cons = 0.12 # Sabit tüketim
            slope_cons = max(0, h_diff) * 1.5 # Eğim tüketimi
            net_power = solar_gain - (base_cons + slope_cons)
            
            battery = min(BATTERY_CAPACITY, battery + net_power)
            total_dist += d_p * PIXEL_TO_METER
            step += 1
            history.append(current_pos)

            for s in science_samples:
                if not s[2] and np.linalg.norm(np.array(current_pos) - np.array(s[:2])) < 2.5:
                    s[2] = True; collected_samples += 1; current_path = []
                    add_log(f"Örnek {collected_samples} Toplandı!")

            if step % 1 == 0:
                disp = raw_img.copy()
                disp[seen_mask == 0] *= 0.4
                img_obj.set_data(disp)
                h_overlay = np.zeros((DEM_SIZE, DEM_SIZE, 4))
                h_overlay[known_hazards == 1] = [1, 0, 0, 0.4]
                hazard_obj.set_data(h_overlay)
                h_arr = np.array(history)
                path_line.set_data(h_arr[:, 1], h_arr[:, 0])
                if current_path:
                    p_arr = np.array(current_path)
                    plan_line.set_data(p_arr[:, 1], p_arr[:, 0])
                rover_marker.set_offsets([current_pos[1], current_pos[0]])
                s_coords = np.array([s[:2] for s in science_samples])
                s_colors = ['#f1c40f' if not s[2] else '#34495e' for s in science_samples]
                samples_marker.set_offsets(s_coords[:, [1, 0]])
                samples_marker.set_facecolors(s_colors)
                
                # --- TELEMETRİ GÜNCELLEME ---
                elapsed_time = int(time.time() - mission_start_time)
                batt_perc = (battery / BATTERY_CAPACITY) * 100
                pwr_status = "(+) ŞARJ" if net_power > 0 else "(-) DEŞARJ"
                log_str = "\n".join(mission_log[-4:])
                
                tel_content = f"""[ MISSION CONTROL ]
------------------------
DURUM: {status}
SÜRE : {elapsed_time} s
KONUM: X:{current_pos[1]*0.5:.1f}m Y:{current_pos[0]*0.5:.1f}m
------------------------
BATARYA: {battery:.1f} Wh (%{batt_perc:.1f})
GÜÇ AKI: {net_power:+.2f} W [{pwr_status}]
------------------------
EĞİM   : {slope:.1f}°
ÖRNEK  : {collected_samples}/3
MESAFE : {total_dist:.1f} m
------------------------
SON OLAYLAR:
{log_str}
------------------------"""
                telemetry_text.set_text(tel_content)
                plt.pause(0.04) 

            if current_pos == final_goal and collected_samples == 3: 
                add_log("Görev Başarıyla Tamamlandı!")
                break
        else:
            add_log("Kritik: Yol Planlanamadı!")
            plt.pause(0.5); break

    if not reset_requested:
        while not reset_requested: plt.pause(0.1)