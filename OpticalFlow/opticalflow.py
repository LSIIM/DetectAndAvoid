import cv2
import numpy as np
import time
import skfuzzy as fuzz
import random
import sys

class TrackedPoint:
    """Represents a tracked point with unique ID"""
    _next_id = 0  # Global counter for unique IDs
    
    @classmethod
    def get_next_id(cls):
        """Get next unique ID"""
        current_id = cls._next_id
        cls._next_id += 1
        return current_id
    
    @classmethod
    def reset_counter(cls):
        """Reset ID counter (useful for testing)"""
        cls._next_id = 0
    
    def __init__(self, position):
        self.id = TrackedPoint.get_next_id()
        self.position = position
    
    def __repr__(self):
        return f"TrackedPoint(id={self.id}, pos={self.position})"

class OpticalFlowContext:
    """Context class to store optical flow state"""
    def __init__(self):
        self.number_clusters = 5
        self.fps = 30
        self.processing_size = (640, 480)
        self.max_point = 30
        self.max_path_length = 20
        self.debug = False
        
        # Feature detection parameters
        self.feature_params = dict(
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
        self.paths = []  # Legacy - will be replaced
        self.frame_iter = 1
        self.previous_centroids = None
        self.cluster_id_mapping = None
        self.previous_u = None  # Store previous fuzzy membership matrix
        self.previous_n_points = 0  # Store number of points from previous clustering
        self.colors = None
        self.start_time = None
        
        # ID-based tracking dictionaries
        self.tracked_points = {}  # {point_id: TrackedPoint}
        self.point_paths = {}  # {point_id: [positions]}
        self.point_positions = {}  # {point_id: [absolute_positions]}
        self.point_clusters = {}  # {point_id: cluster_id}
        self.point_historic_uv = {}  # {point_id: np.ndarray([vx, vy])}
        
        # Detection mask for avoiding problematic regions
        self.detection_mask = None
        self.mask_recovery_rate = 1  # Value added per frame to recover regions
        self.invalid_region_radius = 20  # Radius around invalid points to block

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

def setup(max_point=30, number_clusters=8):
    """Setup optical flow processing"""
    context = OpticalFlowContext()
    context.max_point = max_point
    context.number_clusters = number_clusters

    # Generate colors for clusters
    context.colors = generate_random_colors(max(10, context.number_clusters))
    
    context.start_time = time.perf_counter()
    
    return context

def initialize_tracking(frame, context):
    """Initialize tracking on first frame"""
    if frame is None:
        return False
        
    
    context.old_gray = frame.copy()
    
    # Initialize detection mask (255 = allowed, 0 = blocked)
    if context.detection_mask is None:
        context.detection_mask = np.ones(context.old_gray.shape, dtype=np.uint8) * 255
    
    # Create binary mask for goodFeaturesToTrack (only 0 and 255)
    binary_mask = (context.detection_mask > 127).astype(np.uint8) * 255
    #cv2.rectangle(binary_mask, (0, int(0.82 * binary_mask.shape[0])), (binary_mask.shape[1], binary_mask.shape[0]), 0, -1)
    
    # Find initial features
    context.p0 = cv2.goodFeaturesToTrack(context.old_gray, mask=binary_mask, **context.feature_params, maxCorners=context.max_point)
    
    if context.p0 is None:
        return False
    
    # Initialize mask
    context.mask = np.zeros_like(frame)
    
    # Create tracked points with unique IDs
    context.tracked_points.clear()
    context.point_paths.clear()
    context.point_positions.clear()
    context.point_clusters.clear()
    context.point_historic_uv.clear()
    
    for point in context.p0.reshape(-1, 2):
        tracked_pt = TrackedPoint(point)
        context.tracked_points[tracked_pt.id] = tracked_pt
        context.point_paths[tracked_pt.id] = []
        context.point_positions[tracked_pt.id] = [point.copy()]
        context.point_clusters[tracked_pt.id] = 0
        context.point_historic_uv[tracked_pt.id] = np.zeros(2, dtype=np.float32)
    
    return True

def process_frame(frame, context):
    """Process a single frame with optical flow"""

    # Reprocess good features after certain number of frames
    if context.p0 is not None and (context.frame_iter % int(0.5 * context.fps) == 0) and len(context.p0) < context.max_point:
        # Get current good points as starting point
        tmp_p = context.p0.reshape(-1, 2).copy()
        
        # Create binary mask for goodFeaturesToTrack (only 0 and 255)
        binary_mask = (context.detection_mask > 200).astype(np.uint8) * 255

        #Create a mask to avoid adding new points too close to existing ones
        for existing_pt in tmp_p:
            x, y = int(existing_pt[0]), int(existing_pt[1])
            cv2.circle(binary_mask, (x, y), 4, 0, -1)  # Block a radius of 4 pixels around existing points

        # black bar on bottom 18% 
        #cv2.rectangle(binary_mask, (0, int(0.82 * binary_mask.shape[0])), (binary_mask.shape[1], binary_mask.shape[0]), 0, -1)

        # Find new feature points using detection mask
        new_features = cv2.goodFeaturesToTrack(context.old_gray, mask=binary_mask, **context.feature_params, maxCorners=context.max_point - len(context.p0))
        
        if new_features is not None:
            new_features = new_features.reshape(-1, 2)
            
            # Add new features that are not too close to existing ones
            for new_pt in new_features:
                add_point = True
                for existing_pt in context.p0.reshape(-1, 2):
                    # Check distance (distance squared < 16)
                    if np.sum((existing_pt - new_pt)**2) < 16:
                        add_point = False
                        break
                
                if add_point:
                    # Create new tracked point with unique ID
                    tracked_pt = TrackedPoint(new_pt)
                    context.tracked_points[tracked_pt.id] = tracked_pt
                    context.point_paths[tracked_pt.id] = [[new_pt[0], new_pt[1]]]
                    context.point_positions[tracked_pt.id] = [new_pt.copy()]
                    context.point_clusters[tracked_pt.id] = 0
                    context.point_historic_uv[tracked_pt.id] = np.zeros(2, dtype=np.float32)
                    # Add to p0 for next iteration
                    tmp_p = np.vstack([tmp_p, new_pt]) if len(tmp_p) > 0 else new_pt.reshape(-1, 2)
                    # assign closest cluster to new point and add path = to the closest cluster
                    closest_cluster = 0
                    min_distance = float('inf')
                    for cluster_id, centroid in enumerate(context.previous_centroids if context.previous_centroids is not None else []):
                        distance = np.sqrt((new_pt[0] - centroid[0])**2 + (new_pt[1] - centroid[1])**2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_cluster = cluster_id
                    context.point_clusters[tracked_pt.id] = closest_cluster
                    if context.previous_centroids is not None and closest_cluster < len(context.previous_centroids):
                        centroid = context.previous_centroids[closest_cluster]
                        context.point_historic_uv[tracked_pt.id] = centroid[2:4]

            
            # Update p0 for next iteration
            context.p0 = tmp_p.reshape(-1, 1, 2)

    if frame is None:
        return None, None, None, None
    
    # Check if frame is gray or color
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame_gray = frame.copy()
    else:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
    # Initialize tracking if needed
    if context.p0 is None or context.old_gray is None:
        if not initialize_tracking(frame_gray, context):
            return None, None, None, None
        return None, None, None, None
        
    # Get current tracked point IDs and positions
    point_ids = list(context.tracked_points.keys())
    if len(point_ids) == 0:
        initialize_tracking(frame_gray, context)
        return None, None, None, None
    
    # Build p0 array from tracked points
    context.p0 = np.array([context.tracked_points[pid].position for pid in point_ids]).reshape(-1, 1, 2)
    
    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(context.old_gray, frame_gray, context.p0, None, **context.lk_params)
    
    # Map results back to point IDs
    good_ids = []
    good_new = []
    good_old = []
    
    for i, (pid, status) in enumerate(zip(point_ids, st.flatten())):
        if status == 1:
            good_ids.append(pid)
            good_new.append(p1[i][0])
            good_old.append(context.p0[i][0])
    
    # Remove lost points
    for pid in point_ids:
        if pid not in good_ids:
            del context.tracked_points[pid]
            del context.point_paths[pid]
            if pid in context.point_clusters:
                del context.point_clusters[pid]
    
    good_new = np.array(good_new)
    good_old = np.array(good_old)
    
    if len(good_new) == 0:
        # Reinitialize if no good points
        initialize_tracking(frame, context)
        return None, None, None, None
    
    uvs = (good_new - good_old) #* context.fps  # Convert to velocity in pixels per second

    # Update paths and positions for each tracked point
    for i, pid in enumerate(good_ids):
        # Update position
        context.tracked_points[pid].position = good_new[i]
        
        # Update path
        context.point_paths[pid].append(uvs[i])
        context.point_positions[pid].append(good_new[i].copy())

        #update historic uvs
        current_uv = np.asarray(uvs[i], dtype=np.float32)
        memory_rate = 0.95
        context.point_historic_uv[pid] = [context.point_historic_uv[pid][0]*memory_rate + uvs[i][0]*(1-memory_rate), context.point_historic_uv[pid][1]*memory_rate + current_uv[1]*(1-memory_rate)]#0.99 * np.asarray(context.point_historic_uv[pid], dtype=np.float32) + 0.1 * current_uv
        # Keep only recent path history
        if len(context.point_paths[pid]) > context.max_path_length:
            context.point_paths[pid] = context.point_paths[pid][-context.max_path_length:]

    if len(good_ids) > 0:
        previous_uvs = np.array([
            context.point_paths[pid][-2] if len(context.point_paths[pid]) >= 2 else context.point_paths[pid][-1]
            for pid in good_ids
        ], dtype=np.float32)
        duvs = uvs - previous_uvs
    else:
        duvs = uvs.copy()
        print("Warning: No point paths available for duvs calculation. Using uvs as fallback.")

    # Update for next iteration
    context.old_gray = frame_gray.copy()
    context.p0 = good_new.reshape(-1, 1, 2)
    context.mask = np.zeros_like(frame)  # Reset mask
    context.frame_iter += 1

    # print(f"Tracked points: {len(context.tracked_points)}")
    
    return good_new, good_ids, uvs, duvs

def exclude_invalid_points(context, good_new, good_ids, uvs, duvs, debug=False):
    # Verify tracking consistency - remove points with erratic movement
    valid_ids = []
    invalid_ids = []
    min_frames_check = 15  # Number of frames to analyze
    inconsistency_threshold = 0.95  # Most transitions must be bad to invalidate
    mean_uv = np.mean(uvs, axis=0) if len(uvs) > 0 else np.array([0, 0])

    for i, pid in enumerate(good_ids):
        is_valid = True
        
        # Need at least min_frames_check frames of absolute position history to check consistency
        if pid in context.point_positions and len(context.point_positions[pid]) >= min_frames_check:
            recent_positions = np.asarray(context.point_positions[pid][-min_frames_check:], dtype=np.float32)  # Last N positions
            
            # Calculate velocities from consecutive positions
            recent_vels = []
            for j in range(len(recent_positions) - 1):
                vel = (recent_positions[j + 1] - recent_positions[j]) * context.fps
                recent_vels.append(vel)
            
            if len(recent_vels) < 2:
                # Not enough velocities to check consistency
                valid_ids.append(pid)
                continue
            
            invalid_transitions = 0
            total_transitions = 0
            
            # Check all consecutive velocity pairs
            for j in range(len(recent_vels) - 1):
                vel1 = recent_vels[j]
                vel2 = recent_vels[j + 1]
                
                # Calculate velocity magnitudes
                mag1 = np.linalg.norm(vel1)
                mag2 = np.linalg.norm(vel2)
                
                # Only check if velocities are significant (not stationary)
                if mag1 > 36 and mag2 > 36:
                    total_transitions += 1
                    
                    # Calculate angle difference between consecutive velocities
                    dot_product = np.dot(vel1, vel2)
                    cos_angle = dot_product / (mag1 * mag2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_diff = np.arccos(cos_angle)

                    # #Take into account the mean_uv direction
                    # mean_uv_dot = np.dot(vel1, mean_uv)
                    # mean_uv_mag = np.linalg.norm(mean_uv)
                    # if mean_uv_mag > 0:
                    #     mean_uv_angle = np.arccos(np.clip(mean_uv_dot / (mag1 * mean_uv_mag), -1.0, 1.0))
                    #     angle_diff += mean_uv_angle

                    # Check for sudden direction change (> 90 degrees)
                    if angle_diff > np.pi / 2:
                        invalid_transitions += 1
                        continue
                    
                    # Check for sudden magnitude change (> 5x increase/decrease)
                    mag_ratio = max(mag1, mag2) / (min(mag1, mag2) + 1e-6)
                    if mag_ratio > 5.0:
                        invalid_transitions += 1
            
            # Mark as invalid only if most transitions are inconsistent
            if total_transitions > len(recent_vels)/2:
                inconsistency_ratio = invalid_transitions / total_transitions
                if inconsistency_ratio >= inconsistency_threshold:
                    is_valid = False
        
        if is_valid:
            valid_ids.append(pid)
        else:
            invalid_ids.append(pid)
            # Mark invalid point region in detection mask
            point_pos = good_new[i].astype(int)
            cv2.circle(context.detection_mask, tuple(point_pos), context.invalid_region_radius, 0, -1)
    
    # Remove invalid points from tracking
    for pid in invalid_ids:
        if pid in context.tracked_points:
            del context.tracked_points[pid]
        if pid in context.point_paths:
            del context.point_paths[pid]
        if pid in context.point_positions:
            del context.point_positions[pid]
        if pid in context.point_clusters:
            del context.point_clusters[pid]
    
    # Filter out invalid points from current frame arrays
    if len(valid_ids) < len(good_ids):
        valid_indices = [i for i, pid in enumerate(good_ids) if pid in valid_ids]
        good_ids = valid_ids
        good_new = good_new[valid_indices]
        uvs = uvs[valid_indices]
        duvs = duvs[valid_indices]
        if len(good_new) == 0:
            # All points were invalid, reinitialize
            initialize_tracking(context.old_gray, context)
            return None, None, None, None
    
    context.p0 = good_new.reshape(-1, 1, 2)
    # Gradually recover detection mask (add value to all pixels)
    context.detection_mask = np.clip(context.detection_mask.astype(np.int16) + context.mask_recovery_rate, 
                                        0, 255).astype(np.uint8)

    return good_new, good_ids, uvs, duvs

def cluster_points(context, good_new, good_ids, uvs, duvs, frame, debug=False):

    # Calculate average path vectors for clustering
    path_vectors = []
    for pid in good_ids:
        path = context.point_paths[pid]
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
        # Prepare data for clustering (x, y coordinates + current velocity + historic velocity)
        historic_uvs = np.array([np.asarray(context.point_historic_uv[pid], dtype=np.float32) for pid in good_ids], dtype=np.float32)
        # Stack: [x_positions, y_positions, velocity_x, velocity_y, historic_vx, historic_vy]
        alldata = np.vstack((good_new[:, 0], good_new[:, 1], uvs[:, 0], uvs[:, 1], historic_uvs[:, 0], historic_uvs[:, 1], duvs[:, 0], duvs[:, 1]))

        #normalize different components to avoid scale issues
        alldata[0] = (alldata[0] - np.min(alldata[0])) / (np.max(alldata[0]) - np.min(alldata[0]) + 1e-6)
        alldata[1] = (alldata[1] - np.min(alldata[1])) / (np.max(alldata[1]) - np.min(alldata[1]) + 1e-6)
        alldata[2] = (alldata[2] - np.min(alldata[2])) / (np.max(alldata[2]) - np.min(alldata[2]) + 1e-6)
        alldata[3] = (alldata[3] - np.min(alldata[3])) / (np.max(alldata[3]) - np.min(alldata[3]) + 1e-6)
        alldata[4] = (alldata[4] - np.min(alldata[4])) / (np.max(alldata[4]) - np.min(alldata[4]) + 1e-6)
        alldata[5] = (alldata[5] - np.min(alldata[5])) / (np.max(alldata[5]) - np.min(alldata[5]) + 1e-6)
        alldata[6] = (alldata[6] - np.min(alldata[6])) / (np.max(alldata[6]) - np.min(alldata[6]) + 1e-6)
        alldata[7] = (alldata[7] - np.min(alldata[7])) / (np.max(alldata[7]) - np.min(alldata[7]) + 1e-6)

        try:
            # Perform fuzzy c-means clustering with 50 iterations
            # Use previous membership matrix for warm start if available and point count matches
            init_u = context.previous_u if (context.previous_u is not None and context.previous_n_points == len(good_new)) else None
            
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                alldata, context.number_clusters, 2, error=0.005, maxiter=50, init=init_u)
            
            # Store current membership matrix and point count for next frame
            context.previous_u = u.copy()
            context.previous_n_points = len(good_new)
            
            # Get cluster membership for each point
            raw_membership = np.argmax(u, axis=0)
            
            # Detect outliers based on fuzzy membership strength
            membership_threshold = 0.33  # Minimum membership value to belong to a cluster
            max_memberships = np.max(u, axis=0)  # Maximum membership value for each point
            
            # Mark points with low membership as outliers (-1)
            outlier_mask = max_memberships < membership_threshold
            
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
                        duv_dist = np.linalg.norm(curr_center[6:8] - prev_center[6:8])
                        # Combined distance with weights
                        distances[i, j] = 0.4 * pos_dist + 0.2 * vel_dist + 0.2 * path_dist  + 0.2 * duv_dist
                
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
                
                # Remap cluster membership, marking outliers as -1
                cluster_membership = []
                for i in range(len(raw_membership)):
                    if outlier_mask[i]:
                        cluster_membership.append(-1)  # Outlier
                    else:
                        cluster_membership.append(context.cluster_id_mapping[raw_membership[i]])
            else:
                # First frame - initialize mapping
                context.cluster_id_mapping = {i: i for i in range(context.number_clusters)}
                cluster_membership = []
                for i in range(len(raw_membership)):
                    if outlier_mask[i]:
                        cluster_membership.append(-1)  # Outlier
                    else:
                        cluster_membership.append(raw_membership[i])
            
            # Update previous centroids
            context.previous_centroids = cntr.copy()
            
            # Store cluster membership by ID (-1 for outliers)
            for i, pid in enumerate(good_ids):
                context.point_clusters[pid] = cluster_membership[i]

            #draw cluster and points on path vector space
            pathSpace = np.zeros((800, 1000, 3), dtype=np.uint8)
            cv2.line(pathSpace, (500, 0), (500, 800), (255, 255, 255), 1)  # Vertical center line
            cv2.line(pathSpace, (0, 400), (1000, 400), (255, 255, 255), 1)  # Horizontal center line
            for i, pid in enumerate(good_ids):
                if context.point_clusters[pid] != -1:
                    color = context.colors[context.point_clusters[pid] % len(context.colors)]
                else:
                    color = (128, 128, 128)  # Gray for outliers
                pos = np.asarray(context.point_historic_uv[pid], dtype=np.float32).astype(int) * 5
                cv2.circle(pathSpace,(int(pos[0] + 500), int(pos[1] + 400)), 5, color, -1)
            
            #clusters
            for i, centroid in enumerate(cntr):
                if i < len(context.colors):
                    mapped_id = context.cluster_id_mapping.get(i, i) if context.cluster_id_mapping else i
                    color = context.colors[mapped_id % len(context.colors)]
                    historic_values = np.array(list(context.point_historic_uv.values()), dtype=np.float32)
                    unnormalized_centroid = centroid[4:6] * (np.max(historic_values, axis=0) - np.min(historic_values, axis=0) + 1e-6) + np.min(historic_values, axis=0)

                    center_pos = (int(unnormalized_centroid[0]*5 + 500), int(unnormalized_centroid[1]*5 + 400))
                    cv2.circle(pathSpace, center_pos, 8, color, -1)
                    cv2.circle(pathSpace, center_pos, 8, (50, 50, 50), 1)  # Black border for visibility
            #cv2.imshow("Path Vector Space", pathSpace)

            #draw cluster and points on velocity space
            velSpace = np.zeros((800, 1000, 3), dtype=np.uint8)
            cv2.line(velSpace, (500, 0), (500, 800), (255, 255, 255), 1)  # Vertical center line
            cv2.line(velSpace, (0, 400), (1000, 400), (255, 255, 255), 1)  # Horizontal center line
            for i, pid in enumerate(good_ids):
                if context.point_clusters[pid] != -1:
                    color = context.colors[context.point_clusters[pid] % len(context.colors)]
                else:
                    color = (128, 128, 128)  # Gray for outliers
                pos = uvs[i].astype(int) * 5
                cv2.circle(velSpace,(int(pos[0] + 500), int(pos[1] + 400)), 5, color, -1)
            #clusters
            for i, centroid in enumerate(cntr):
                if i < len(context.colors):
                    mapped_id = context.cluster_id_mapping.get(i, i) if context.cluster_id_mapping else i
                    color = context.colors[mapped_id % len(context.colors)]
                    unnormalized_centroid = centroid[2:4] * (np.max(uvs, axis=0) - np.min(uvs, axis=0) + 1e-6) + np.min(uvs, axis=0)

                    center_pos = (int(unnormalized_centroid[0]*5 + 500), int(unnormalized_centroid[1]*5 + 400))
                    cv2.circle(velSpace, center_pos, 8, color, -1)
                    cv2.circle(velSpace, center_pos, 8, (50, 50, 50), 1)  # Black border for visibility

            #cv2.imshow("Velocity Space", velSpace)

        except:
            # Fallback if clustering fails
            cluster_membership = [0] * len(good_new)
            for pid in good_ids:
                context.point_clusters[pid] = 0
    else:
        # Not enough points for clustering, assign all to cluster 0
        cluster_membership = [0] * len(good_new)
        for pid in good_ids:
            context.point_clusters[pid] = 0

    # print clusters distances from each other
    # for c in cntr:
    #     print(f"Cluster center at ({c[0]:.1f}, {c[1]:.1f}) with velocity ({c[2]:.1f}, {c[3]:.1f})")

    # Draw the tracks with cluster colors
    #result_frame = frame_resized.copy()
    for i, pid in enumerate(good_ids):
        new = good_new[i]
        a, b = int(new[0]), int(new[1])
        
        # Use cluster color or gray for outliers
        cluster_id = context.point_clusters.get(pid, 0)
        if cluster_id == -1:
            # Outlier - use gray color
            color = (128, 128, 128)
        else:
            color = context.colors[cluster_id % len(context.colors)]
        
        # Draw current point
        #result_frame = cv2.circle(result_frame, (a, b), 5, color, -1)
        
        # Draw path if it has points
        path = context.point_paths[pid]
        if len(path) > 0:
            pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(context.mask, [pts], False, color, 2)
        
        # Optional: Draw point ID for debugging
        # if context.debug:
        #     cv2.putText(result_frame, str(pid), (a + 8, b - 8), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


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
                # vel_magnitude = np.sqrt(centroid[2]**2 + centroid[3]**2)
                # vel_text = f"V:{vel_magnitude:.1f}"
                
                # Display current velocity
                # cv2.putText(result_frame, vel_text, (center_pos[0] + 15, center_pos[1] - 15), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # cv2.putText(result_frame, vel_text, (center_pos[0] + 15, center_pos[1] - 15), 
                        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                # Draw centroid circle
                # cv2.circle(result_frame, center_pos, 8, color, -1)
                # Draw velocity vector
                # end_pos = (int(centroid[0] + centroid[2] * 0.1), int(centroid[1] + centroid[3] * 0.1))
                # cv2.arrowedLine(result_frame, center_pos, end_pos, color, 2, tipLength=0.3)

    debug_frame = np.ones(frame.shape, dtype=np.uint8) * 255
    roi_list = []
    
    # Group points by cluster (excluding outliers)
    cluster_points = {}
    for pid, cluster_id in context.point_clusters.items():
        if pid in context.tracked_points and cluster_id != -1:  # Exclude outliers
            if cluster_id not in cluster_points:
                cluster_points[cluster_id] = []
            cluster_points[cluster_id].append(context.tracked_points[pid].position)
    
    for i, centroid in enumerate(context.previous_centroids if context.previous_centroids is not None else []):
        if i < len(context.colors):
            # Get mapped cluster ID
            mapped_id = context.cluster_id_mapping.get(i, i) if context.cluster_id_mapping else i
            color = context.colors[mapped_id % len(context.colors)]
            
            # Extract and paste ROI containing all cluster points
            if mapped_id in cluster_points and len(cluster_points[mapped_id]) > 0:
                points = np.array(cluster_points[mapped_id])
                
                # Calculate bounding box with margin
                margin = 20
                x_min = max(0, int(np.min(points[:, 0])) - margin)
                x_max = min(frame.shape[1], int(np.max(points[:, 0])) + margin)
                y_min = max(0, int(np.min(points[:, 1])) - margin)
                y_max = min(frame.shape[0], int(np.max(points[:, 1])) + margin)
                
                # Extract ROI from result frame
                if x_max > x_min and y_max > y_min:
                    roi = frame[y_min:y_max, x_min:x_max].copy()
                    
                    # Draw border around ROI
                    cv2.rectangle(roi, (0, 0), (roi.shape[1]-1, roi.shape[0]-1), color, 2)
                    
                    # Paste ROI at its original position in debug_frame
                    if debug:
                        debug_frame[y_min:y_max, x_min:x_max] = roi

                    roi_list.append([x_min, y_min, x_max, y_max])

    if debug:
        cv2.imshow("Debug Mask", debug_frame)

    return roi_list

def cleanup(context):
    """Cleanup optical flow resources"""
    if context:
        context.p0 = None
        context.old_gray = None
        context.mask = None
        context.paths = []
        context.tracked_points.clear()
        context.point_paths.clear()
        context.point_positions.clear()
        context.point_clusters.clear()
        TrackedPoint.reset_counter()

# Standalone execution for testing
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        print("No video path provided.")
        sys.exit(1)

    # Open video file
    capture = cv2.VideoCapture(filename)
    if not capture.isOpened():
        print(f"Error: Could not open video {filename}")
        sys.exit(1)

    fps = capture.get(cv2.CAP_PROP_FPS)
    
    # Setup optical flow with max points
    context = setup(100, 8)
    ksize = 3
    n_prewitt = 3
    degree = 1
    delta = 0

    
    print(f"Processing video: {filename}")
    print(f"FPS: {fps}")

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (853, 480))
    
        # Process frame
        good_new, good_ids, uvs, duvs = process_frame(frame, context)

        # Exclude invalid points based on tracking consistency
        if good_new is not None and good_ids is not None and uvs is not None:
            good_new, good_ids, uvs, duvs = exclude_invalid_points(context, good_new, good_ids, uvs, duvs)

        # Cluster points and visualize
        if good_new is not None and good_ids is not None and uvs is not None:
            cluster_windows = cluster_points(context, good_new, good_ids, uvs, duvs, frame)

        # draw uvs on the good_new points
        for i, pid in enumerate(good_ids) if good_new is not None else []:
            new = good_new[i]
            a, b = int(new[0]), int(new[1])
            u, v = uvs[i]

            if context.point_clusters[pid] != -1:
                color = context.colors[context.point_clusters[pid] % len(context.colors)]
            else:
                color = (128, 128, 128)  # Gray for outliers

            # Draw arrow for optical flow
            frame = cv2.circle(frame, (a, b), 5, color, -1)
            frame = cv2.arrowedLine(frame, (a, b), (int(a + u), int(b + v)), color, 2, tipLength=0.2)

        # Display result
        cv2.imshow("Optical Flow", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
    
    # Cleanup
    capture.release()
    cv2.destroyAllWindows()
    cleanup(context)