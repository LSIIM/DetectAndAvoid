import cv2
import numpy as np
import time
import skfuzzy as fuzz
import random

number_clusters = 5 
filename = "C:/Users/dudu1/Desktop/drone.mp4"
capture = cv2.VideoCapture(filename)

frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)
resize_scale = 650.0/frame_height
max_point = 100

ret, old_frame = capture.read()
old_frame = cv2.resize(old_frame, (0,0), fx=resize_scale, fy=resize_scale)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

feature_params = dict( maxCorners = max_point,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

mask = np.zeros_like(old_frame)

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Generate random colors for clusters
def generate_random_colors(n_colors):
    """Generate n_colors random colors with good visibility"""
    colors = []
    random.seed(42)  
    for i in range(n_colors):
        r = random.uniform(0, 255)
        g = random.uniform(0, 255)
        b = random.uniform(0, 255)
        colors.append((b, g, r))  
    return colors

colors = generate_random_colors(max(number_clusters, 10))  # Generate at least 10 colors

# Variables for cluster tracking
previous_centroids = None
cluster_id_mapping = None

# Variables for path tracking
paths = []  
max_path_length = 10  

start_time = time.perf_counter() 


for iter in range(100):
  ret, frame = capture.read()
  if not ret:
    break

  frame = cv2.resize(frame, (0,0), fx=resize_scale, fy=resize_scale)
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # calculate optical flow
  p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

  # Select good points using boolean indexing 
  good_new = p1[st==1]
  good_old = p0[st==1]

  # Vectorized velocity calculation
  uvs = (good_new - good_old) * fps

  # Update paths for each point
  if len(paths) == 0:
    paths = [[] for _ in range(len(good_new))]
  
  # Adjust paths array size to match current points
  if len(paths) != len(good_new):
    if len(paths) > len(good_new):
      # Remove excess paths (points were lost)
      paths = paths[:len(good_new)]
    else:
      # Add new paths (new points detected)
      paths.extend([[] for _ in range(len(good_new) - len(paths))])
  
  # Update each point's path with current velocity
  for i in range(len(good_new)):
    if i < len(uvs):
      paths[i].append(uvs[i])
      # Keep only recent path history
      if len(paths[i]) > max_path_length:
        paths[i] = paths[i][-max_path_length:]
  
  # Calculate average path vectors for clustering
  path_vectors = []
  for path in paths:
    if len(path) > 0:
      # Calculate average velocity over the path
      avg_path = np.mean(path, axis=0)
      path_vectors.append(avg_path)
    else:
      path_vectors.append(np.array([0.0, 0.0]))
  
  path_vectors = np.array(path_vectors)

  # Perform fuzzy clustering if we have enough points
  cluster_membership = []
  if len(good_new) >= number_clusters:
    # Prepare data for clustering (x, y coordinates + current velocity + path average)
    # Stack: [x_positions, y_positions, velocity_x, velocity_y, path_avg_x, path_avg_y]
    alldata = np.vstack((good_new[:, 0], good_new[:, 1], uvs[:, 0], uvs[:, 1], path_vectors[:, 0], path_vectors[:, 1]))
    
    try:
      # Perform fuzzy c-means clustering with 50 iterations
      cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
          alldata, number_clusters, 2, error=0.005, maxiter=50, init=None)
      
      # Get cluster membership for each point
      raw_membership = np.argmax(u, axis=0)
      
      # Handle cluster tracking for consistent colors
      if previous_centroids is not None:
        # Calculate distances between current and previous centroids
        # Consider position, current velocity, and path average
        distances = np.zeros((len(cntr), len(previous_centroids)))
        for i, curr_center in enumerate(cntr):
          for j, prev_center in enumerate(previous_centroids):
            # Weight position, current velocity, and path components
            pos_dist = np.linalg.norm(curr_center[:2] - prev_center[:2])  # Position distance
            vel_dist = np.linalg.norm(curr_center[2:4] - prev_center[2:4])  # Current velocity distance
            path_dist = np.linalg.norm(curr_center[4:6] - prev_center[4:6])  # Path average distance
            # Combined distance with weights (you can adjust these weights)
            distances[i, j] = 0.5 * pos_dist + 0.25 * vel_dist + 0.25 * path_dist
        
        # Create mapping using Hungarian algorithm (simplified greedy approach)
        used_prev_ids = set()
        new_mapping = {}
        
        # Sort by distance and assign closest matches first
        flat_indices = np.unravel_index(np.argsort(distances, axis=None), distances.shape)
        for curr_id, prev_id in zip(flat_indices[0], flat_indices[1]):
          if curr_id not in new_mapping and prev_id not in used_prev_ids:
            new_mapping[curr_id] = cluster_id_mapping[prev_id] if cluster_id_mapping is not None else prev_id
            used_prev_ids.add(prev_id)
        
        # Assign remaining clusters to unused IDs
        available_ids = set(range(number_clusters)) - set(new_mapping.values())
        for curr_id in range(len(cntr)):
          if curr_id not in new_mapping:
            if available_ids:
              new_mapping[curr_id] = available_ids.pop()
            else:
              new_mapping[curr_id] = curr_id
        
        cluster_id_mapping = new_mapping
        
        # Remap cluster membership
        cluster_membership = [cluster_id_mapping[raw_membership[i]] for i in range(len(raw_membership))]
      else:
        # First frame - initialize mapping
        cluster_id_mapping = {i: i for i in range(number_clusters)}
        cluster_membership = raw_membership.tolist()
      
      # Update previous centroids
      previous_centroids = cntr.copy()
      
    except:
      # Fallback if clustering fails
      cluster_membership = [0] * len(good_new)
  else:
    # Not enough points for clustering, assign all to cluster 0
    cluster_membership = [0] * len(good_new)

  # Draw the tracks with cluster colors
  for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = int(new[0]), int(new[1])
    c, d = int(old[0]), int(old[1])
    
    # Use cluster color
    cluster_id = cluster_membership[i] if i < len(cluster_membership) else 0
    color = colors[cluster_id % len(colors)]
    
    mask = cv2.line(mask, (a, b), (c, d), color, 2)
    frame = cv2.circle(frame, (a, b), 5, color, -1)

  # Draw cluster centroids and their velocity vectors (if clustering was successful)
  if len(good_new) >= number_clusters and 'cntr' in locals():
    for i, centroid in enumerate(cntr):
      if i < len(colors):
        # Get mapped cluster ID
        mapped_id = cluster_id_mapping.get(i, i) if cluster_id_mapping else i
        color = colors[mapped_id % len(colors)]
        
        # Draw centroid position
        center_pos = (int(centroid[0]), int(centroid[1]))
        cv2.circle(frame, center_pos, 7, color, -1)
        #cv2.circle(frame, center_pos, 10, (255, 255, 255), 1)  # White border
        
        # Draw velocity vector from centroid (current velocity)
        velocity_end = (int(centroid[0] + centroid[2] * 5), int(centroid[1] + centroid[3] * 5))
        cv2.arrowedLine(frame, center_pos, velocity_end, color, 3)
        
        # Draw path vector from centroid (average path direction)
        path_end = (int(centroid[0] + centroid[4] * 8), int(centroid[1] + centroid[5] * 8))
        cv2.arrowedLine(frame, center_pos, path_end, (255, 255, 255), 2)  # White arrow for path
        
        # Display velocity and path magnitudes
        vel_magnitude = np.sqrt(centroid[2]**2 + centroid[3]**2)
        path_magnitude = np.sqrt(centroid[4]**2 + centroid[5]**2)
        vel_text = f"V:{vel_magnitude:.1f}"
        path_text = f"P:{path_magnitude:.1f}"
        
        # Display current velocity
        cv2.putText(frame, vel_text, (center_pos[0] + 15, center_pos[1] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, vel_text, (center_pos[0] + 15, center_pos[1] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Display path average
        cv2.putText(frame, path_text, (center_pos[0] + 15, center_pos[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, path_text, (center_pos[0] + 15, center_pos[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

  img = cv2.add(frame, mask)

  # Calculate and display FPS (optimized)
  end_time = time.perf_counter()
  if (end_time - start_time) > 0:
    processing_fps = 1 / (end_time - start_time)
  else:
    processing_fps = 0
  
  # Display FPS on frame
  fps_text = f"FPS: {int(processing_fps)}"
  cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
  cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

  # Now update the previous frame and previous points
  old_gray = frame_gray.copy()
  p0 = good_new.reshape(-1, 1, 2)

  # Display the resulting frame
  cv2.imshow("teste", img)
  key = cv2.waitKey(1) & 0xFF
  
  if key == 27:  # ESC
    break
  elif key == ord('+') or key == ord('='):  # Increase clusters
    if number_clusters < 10:
      number_clusters += 1
      # Reset cluster tracking when changing number of clusters
      previous_centroids = None
      cluster_id_mapping = None
      print(f"Clusters increased to: {number_clusters}")
  elif key == ord('-'):  # Decrease clusters
    if number_clusters > 2:
      number_clusters -= 1
      # Reset cluster tracking when changing number of clusters
      previous_centroids = None
      cluster_id_mapping = None
      print(f"Clusters decreased to: {number_clusters}")
  elif key == ord('p'):  # Increase path length
    if max_path_length < 50:
      max_path_length += 5
      print(f"Path length increased to: {max_path_length}")
  elif key == ord('o'):  # Decrease path length
    if max_path_length > 5:
      max_path_length -= 5
      print(f"Path length decreased to: {max_path_length}")  

  start_time = end_time # Reuse end_time for better performance


capture.release()