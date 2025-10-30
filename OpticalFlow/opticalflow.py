import cv2
import numpy as np
import time
import skfuzzy as fuzz
import random
import sys

class OpticalFlowContext:
    """Context class to store optical flow state"""
    def __init__(self):
        self.number_clusters = 5
        self.fps = 30
        self.processing_size = (640, 480)
        self.max_point = 100
        self.max_path_length = 10
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=self.max_point,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # State variables
        self.p0 = None
        self.old_gray = None
        self.mask = None
        self.paths = []
        self.frame_iter = 0
        self.previous_centroids = None
        self.cluster_id_mapping = None
        self.colors = None
        self.start_time = None

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

def setup(clusters=5, fps=30, processing_size=(640, 480)):
    """Setup optical flow processing"""
    context = OpticalFlowContext()
    context.number_clusters = clusters
    context.fps = fps
    context.processing_size = processing_size
    
    # Generate colors for clusters
    context.colors = generate_random_colors(max(clusters, 10))
    
    context.start_time = time.perf_counter()
    
    return context

def initialize_tracking(frame, context):
    """Initialize tracking on first frame"""
    if frame is None:
        return False
        
    # Resize frame
    frame_resized = cv2.resize(frame, context.processing_size)
    context.old_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    # Find initial features
    context.p0 = cv2.goodFeaturesToTrack(context.old_gray, mask=None, **context.feature_params)
    
    if context.p0 is None:
        return False
        
    # Initialize mask
    context.mask = np.zeros_like(frame_resized)
    context.paths = []
    
    return True

def process_frame(frame, context):
    """Process a single frame with optical flow"""
    if frame is None:
        return frame
        
    # Resize frame for processing
    frame_resized = cv2.resize(frame, context.processing_size)
    
    # Initialize tracking if needed
    if context.p0 is None or context.old_gray is None:
        if not initialize_tracking(frame, context):
            return frame_resized
        return frame_resized
    
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(context.old_gray, frame_gray, context.p0, None, **context.lk_params)
    
    # Select good points
    good_new = p1[st == 1]
    good_old = context.p0[st == 1]
    
    if len(good_new) == 0:
        # Reinitialize if no good points
        initialize_tracking(frame, context)
        return frame_resized
    
    # Calculate velocities
    uvs = (good_new - good_old) * context.fps
    
    # Update paths for each point
    if len(context.paths) == 0:
        context.paths = [[] for _ in range(len(good_new))]
    
    # Adjust paths array size to match current points
    if len(context.paths) != len(good_new):
        if len(context.paths) > len(good_new):
            # Remove excess paths (points were lost)
            context.paths = context.paths[:len(good_new)]
        else:
            # Add new paths (new points detected)
            context.paths.extend([[] for _ in range(len(good_new) - len(context.paths))])
    
    # Update each point's path with current velocity
    for i in range(len(good_new)):
        if i < len(uvs):
            context.paths[i].append(uvs[i])
            # Keep only recent path history
            if len(context.paths[i]) > context.max_path_length:
                context.paths[i] = context.paths[i][-context.max_path_length:]
    
    # Calculate average path vectors for clustering
    path_vectors = []
    for path in context.paths:
        if len(path) > 0:
            # Calculate average velocity over the path
            avg_path = np.mean(path, axis=0)
            path_vectors.append(avg_path)
        else:
            path_vectors.append(np.array([0.0, 0.0]))
    
    path_vectors = np.array(path_vectors)

    # Perform fuzzy clustering if we have enough points
    cluster_membership = []
    if len(good_new) >= context.number_clusters:
        # Prepare data for clustering (x, y coordinates + current velocity + path average)
        # Stack: [x_positions, y_positions, velocity_x, velocity_y, path_avg_x, path_avg_y]
        alldata = np.vstack((good_new[:, 0], good_new[:, 1], uvs[:, 0], uvs[:, 1], path_vectors[:, 0], path_vectors[:, 1]))
        
        try:
            # Perform fuzzy c-means clustering with 50 iterations
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                alldata, context.number_clusters, 2, error=0.005, maxiter=50, init=None)
            
            # Get cluster membership for each point
            raw_membership = np.argmax(u, axis=0)
            
            # Handle cluster tracking for consistent colors
            if context.previous_centroids is not None:
                # Calculate distances between current and previous centroids
                distances = np.zeros((len(cntr), len(context.previous_centroids)))
                for i, curr_center in enumerate(cntr):
                    for j, prev_center in enumerate(context.previous_centroids):
                        # Weight position, current velocity, and path components
                        pos_dist = np.linalg.norm(curr_center[:2] - prev_center[:2])
                        vel_dist = np.linalg.norm(curr_center[2:4] - prev_center[2:4])
                        path_dist = np.linalg.norm(curr_center[4:6] - prev_center[4:6])
                        # Combined distance with weights
                        distances[i, j] = 0.5 * pos_dist + 0.25 * vel_dist + 0.25 * path_dist
                
                # Create mapping using greedy approach
                used_prev_ids = set()
                new_mapping = {}
                
                # Sort by distance and assign closest matches first
                flat_indices = np.unravel_index(np.argsort(distances, axis=None), distances.shape)
                for curr_id, prev_id in zip(flat_indices[0], flat_indices[1]):
                    if curr_id not in new_mapping and prev_id not in used_prev_ids:
                        new_mapping[curr_id] = context.cluster_id_mapping[prev_id] if context.cluster_id_mapping is not None else prev_id
                        used_prev_ids.add(prev_id)
                
                # Assign remaining clusters to unused IDs
                available_ids = set(range(context.number_clusters)) - set(new_mapping.values())
                for curr_id in range(len(cntr)):
                    if curr_id not in new_mapping:
                        if available_ids:
                            new_mapping[curr_id] = available_ids.pop()
                        else:
                            new_mapping[curr_id] = curr_id
                
                context.cluster_id_mapping = new_mapping
                
                # Remap cluster membership
                cluster_membership = [context.cluster_id_mapping[raw_membership[i]] for i in range(len(raw_membership))]
            else:
                # First frame - initialize mapping
                context.cluster_id_mapping = {i: i for i in range(context.number_clusters)}
                cluster_membership = raw_membership.tolist()
            
            # Update previous centroids
            context.previous_centroids = cntr.copy()
            
        except:
            # Fallback if clustering fails
            cluster_membership = [0] * len(good_new)
    else:
        # Not enough points for clustering, assign all to cluster 0
        cluster_membership = [0] * len(good_new)

    # Draw the tracks with cluster colors
    result_frame = frame_resized.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = int(new[0]), int(new[1])
        c, d = int(old[0]), int(old[1])
        
        # Use cluster color
        cluster_id = cluster_membership[i] if i < len(cluster_membership) else 0
        color = context.colors[cluster_id % len(context.colors)]
        
        context.mask = cv2.line(context.mask, (a, b), (c, d), color, 2)
        result_frame = cv2.circle(result_frame, (a, b), 5, color, -1)

    # Draw cluster centroids and their velocity vectors (if clustering was successful)
    if len(good_new) >= context.number_clusters and 'cntr' in locals():
        for i, centroid in enumerate(cntr):
            if i < len(context.colors):
                # Get mapped cluster ID
                mapped_id = context.cluster_id_mapping.get(i, i) if context.cluster_id_mapping else i
                color = context.colors[mapped_id % len(context.colors)]
                
                # Draw centroid position
                center_pos = (int(centroid[0]), int(centroid[1]))
                
                # Display velocity magnitude
                vel_magnitude = np.sqrt(centroid[2]**2 + centroid[3]**2)
                vel_text = f"V:{vel_magnitude:.1f}"
                
                # Display current velocity
                cv2.putText(result_frame, vel_text, (center_pos[0] + 15, center_pos[1] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(result_frame, vel_text, (center_pos[0] + 15, center_pos[1] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Combine frame with mask
    img = cv2.add(result_frame, context.mask)

    # Calculate and display FPS
    end_time = time.perf_counter()
    if (end_time - context.start_time) > 0:
        processing_fps = 1 / (end_time - context.start_time)
    else:
        processing_fps = 0
    
    # Display FPS on frame
    fps_text = f"FPS: {int(processing_fps)}"
    cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Update for next iteration
    context.old_gray = frame_gray.copy()
    context.p0 = good_new.reshape(-1, 1, 2)

    # Reprocess good features after certain number of frames
    if context.frame_iter >= int(0.5 * context.fps) - 1:
        # Get current good points as starting point
        tmp_p = good_new.copy()
        
        # Find new feature points
        new_features = cv2.goodFeaturesToTrack(context.old_gray, mask=None, **context.feature_params)
        
        if new_features is not None:
            new_features = new_features.reshape(-1, 2)
            
            # Add new features that are not too close to existing ones
            for new_pt in new_features:
                add_point = True
                for existing_pt in good_new:
                    # Check distance (distance squared < 16)
                    if np.sum((existing_pt - new_pt)**2) < 16:
                        add_point = False
                        break
                
                if add_point:
                    # Add to p0 for next iteration
                    tmp_p = np.vstack([tmp_p, new_pt]) if len(tmp_p) > 0 else new_pt.reshape(-1, 2)
            
            # Update p0 for next iteration
            context.p0 = tmp_p.reshape(-1, 1, 2)
        
        # Reset iteration counter and clear tracking data
        context.frame_iter = -1
        context.paths = []  # Clear paths
        context.mask = np.zeros_like(frame_resized)  # Reset mask
    
    context.frame_iter += 1
    context.start_time = end_time
    
    return img

def cleanup(context):
    """Cleanup optical flow resources"""
    if context:
        context.p0 = None
        context.old_gray = None
        context.mask = None
        context.paths = []

# Standalone execution for testing
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python opticalflow.py <clusters> [video_path]")
        print("  <clusters>: Number of clusters (required)")
        print("  [video_path]: Path to video file (optional)")
        sys.exit(1)

    number_clusters = int(sys.argv[1])

    # Use provided video path or default
    if len(sys.argv) >= 3:
        filename = sys.argv[2]
    else:
        filename = "C:\\Users\\dudu1\\Desktop\\DeA\\DetectAndAvoid\\Videos\\fev_corte_2.mp4"

    capture = cv2.VideoCapture(filename)
    if not capture.isOpened():
        print(f"Error: Could not open video {filename}")
        sys.exit(1)

    fps = capture.get(cv2.CAP_PROP_FPS)
    
    # Setup optical flow
    context = setup(clusters=number_clusters, fps=fps)
    
    print(f"Processing video: {filename}")
    print(f"Clusters: {number_clusters}")
    print(f"FPS: {fps}")
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        # Process frame
        result = process_frame(frame, context)
        
        # Display result
        cv2.imshow("Optical Flow", result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
        elif key == ord('+') or key == ord('='):  # Increase clusters
            if context.number_clusters < 10:
                context.number_clusters += 1
                # Reset cluster tracking when changing number of clusters
                context.previous_centroids = None
                context.cluster_id_mapping = None
                print(f"Clusters increased to: {context.number_clusters}")
        elif key == ord('-'):  # Decrease clusters
            if context.number_clusters > 2:
                context.number_clusters -= 1
                # Reset cluster tracking when changing number of clusters
                context.previous_centroids = None
                context.cluster_id_mapping = None
                print(f"Clusters decreased to: {context.number_clusters}")

    # Cleanup
    capture.release()
    cv2.destroyAllWindows()
    cleanup(context)