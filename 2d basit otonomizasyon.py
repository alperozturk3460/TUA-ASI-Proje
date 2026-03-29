import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from PIL import Image, ImageDraw
import random

# --- GLOBAL DEĞİŞKENLER ---
true_obs, known_obstacles, raw_img = None, None, None
reset_requested = False
science_samples = []
collected_samples = 0

def generate_mission():
    global true_obs, raw_img, known_obstacles, science_samples, collected_samples
    img = Image.new('L', (500, 500), color=215)
    draw = ImageDraw.Draw(img)
    for _ in range(25):
        x, y = np.random.randint(50, 450, 2); r = np.random.randint(20, 65)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=np.random.randint(40, 110))
    img_res = img.resize((80, 80))
    raw_img = np.array(img_res)
    true_obs = np.where(raw_img < 50, 999, 1)
    known_obstacles = np.ones((80, 80))
    science_samples = [[np.random.randint(15, 65), np.random.randint(15, 65), False] for _ in range(3)]
    collected_samples = 0

def change_map_callback(event):
    global reset_requested
    reset_requested = True

def on_click(event):
    if event.inaxes == ax_map and event.button == 1:
        ix, iy = int(event.ydata), int(event.xdata)
        true_obs[max(0, ix-2):min(80, ix+2), max(0, iy-2):min(80, iy+2)] = 999

def get_physics_data(pos, prev_pos, img):
    y, x = pos
    brightness = float(img[y, x])
    prev_val = float(img[prev_pos[0], prev_pos[1]])
    slope = abs(brightness - prev_val)
    
    # HIZ: Eğimde yavaşlar
    speed = 0.5 * (1.0 - (slope / 160.0))
    speed = max(0.08, speed) 
    
    # --- SOLAR ENERJİ MODELİ ---
    # Parlak pikseller = Güneş paneli verimi artar
    solar_gain = (brightness / 255.0) * 0.08 
    # Karanlık ve eğimli yerler = Tüketim artar
    base_consumption = 0.04
    slope_penalty = (255 - brightness) * 0.005
    
    net_power = solar_gain - (base_consumption + slope_penalty)
    return speed, net_power, slope

def find_nearest_sample(current_pos, samples):
    # Henüz toplanmamış en yakın örneği bulur
    min_dist = float('inf')
    nearest = None
    for s in samples:
        if not s[2]: # s[2] toplanma durumu
            d = np.linalg.norm(np.array(current_pos) - np.array(s[:2]))
            if d < min_dist:
                min_dist = d
                nearest = tuple(s[:2])
    return nearest

def a_star(known_grid, grid_raw, start, goal):
    rows, cols = known_grid.shape
    open_list = [(0, start)]; came_from, g_score = {}, {start: 0}
    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current); current = came_from[current]
            return path[::-1]
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if known_grid[neighbor[0], neighbor[1]] == 999: continue
                cost = (255 - grid_raw[neighbor[0], neighbor[1]]) * 4.0 + (1.41 if dx!=0 and dy!=0 else 1.0)
                new_g = g_score[current] + cost
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    came_from[neighbor], g_score[neighbor] = current, new_g
                    h = np.sqrt((neighbor[0]-goal[0])**2 + (neighbor[1]-goal[1])**2) * 1.5
                    heapq.heappush(open_list, (new_g + h, neighbor))
    return None

# --- ARAYÜZ VE DÖNGÜ ---
plt.ion()
fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[5, 1])
ax_map = fig.add_subplot(gs[:, 0]); ax_telemetry = fig.add_subplot(gs[0, 1])
ax_telemetry.set_axis_off()

ax_button = plt.axes([0.76, 0.05, 0.20, 0.07])
btn_change = Button(ax_button, 'YENİ GÖREV / RESET', color='#2c3e50', hovercolor='#34495e')
btn_change.label.set_color('white')
btn_change.on_clicked(change_map_callback)
fig.canvas.mpl_connect('button_press_event', on_click)

while True:
    generate_mission()
    reset_requested, current_pos, final_goal = False, (5, 5), (75, 75)
    history, fuel, total_dist, step = [current_pos], 100.0, 0.0, 0

    while True: # Örnek toplama ve hedefe varma döngüsü
        if reset_requested or fuel <= 0: break
        
        # 1. EN YAKIN HEDEFİ SEÇ (TSP Mantığı)
        target = find_nearest_sample(current_pos, science_samples)
        if target is None: target = final_goal # Örnek bittiyse ana hedefe git
        
        while current_pos != target:
            if reset_requested or fuel <= 0: break
            step += 1
            
            # LiDAR Tarama
            y_g, x_g = np.ogrid[:80, :80]
            mask = (y_g - current_pos[0])**2 + (x_g - current_pos[1])**2 <= 14**2
            known_obstacles[mask] = true_obs[mask]
            
            path = a_star(known_obstacles, raw_img, current_pos, target)
            if not path: break
            
            prev_pos, current_pos = current_pos, path[0]
            speed, net_power, slope = get_physics_data(current_pos, prev_pos, raw_img)
            
            history.append(current_pos)
            total_dist += np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
            fuel = min(100.0, fuel + net_power) # Batarya 100'ü geçemez
            
            for s in science_samples:
                if not s[2] and np.linalg.norm(np.array(current_pos) - np.array(s[:2])) < 2.5:
                    s[2] = True; collected_samples += 1; break # Hedef değişeceği için iç döngüden çık
            
            if target in [tuple(s[:2]) for s in science_samples] and any(s[2] for s in science_samples if tuple(s[:2]) == target):
                break # Örnek toplandı, yeni en yakını bulmak için çık

            # GÖRSELLEŞTİRME
            ax_map.clear()
            display_img = raw_img.copy().astype(float)
            display_img[~mask] *= 0.6
            ax_map.imshow(display_img, cmap='gray', origin='upper')
            ax_map.imshow((known_obstacles == 999), alpha=0.3, cmap='Reds')
            h_arr = np.array(history); ax_map.plot(h_arr[:,1], h_arr[:,0], color='#00FF00', linewidth=2)
            for s in science_samples:
                ax_map.scatter(s[1], s[0], color='cyan' if not s[2] else '#2c3e50', s=80, marker='D', edgecolors='white')
            ax_map.scatter(final_goal[1], final_goal[0], color='gold', marker='*', s=350, zorder=10)
            ax_map.scatter(current_pos[1], current_pos[0], color='white', s=130, edgecolors='black', zorder=10)
            
            # Telemetri
            ax_telemetry.clear(); ax_telemetry.set_axis_off()
            charging_status = "SARJ OLUYOR" if net_power > 0 else "DESARJ"
            tel_txt = f"""
[ MISSION CONTROL ]
-------------------
BATARYA : %{max(0, fuel):.1f}
GÜÇ     : {charging_status}
-------------------
HIZ     : {speed:.2f} m/s
ÖRNEK   : {collected_samples}/3
-------------------
MESAFE  : {total_dist:.1f} m
SÜRE    : {step} s
HEDEF   : {target}
-------------------
* Güneşte şarj olur.
* En yakın hedefe gider.
            """
            ax_telemetry.text(0.02, 0.5, tel_txt, fontsize=10, family='monospace', va='center', color='white',
                              bbox=dict(boxstyle="round,pad=0.5", facecolor='#1e1e1e', alpha=0.9))
            ax_map.set_title("Otonom Ay Görevi - Solar & Rota Optimizasyonu", fontsize=14, fontweight='bold')
            plt.pause(0.01 / speed)

        if current_pos == final_goal: break # Görev bitti

    if not reset_requested:
        while not reset_requested: plt.pause(0.1)