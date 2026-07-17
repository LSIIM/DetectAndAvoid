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
        self.frame_iter = 0
        self.previous_centroids = None
        self.cluster_id_mapping = None
        self.previous_u = None  # Store previous fuzzy membership matrix
        self.previous_n_points = 0  # Store number of points from previous clustering
        self.colors = None
        self.start_time = None
        
        # ID-based tracking dictionaries
        self.tracked_points = {}  # {point_id: TrackedPoint}
        self.point_paths = {}  # {point_id: [positions]}
        self.point_clusters = {}  # {point_id: cluster_id}
        
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

def setup(max_point=30):
    """Setup optical flow processing"""
    context = OpticalFlowContext()
    context.max_point = max_point
    
    # Generate colors for clusters
    context.colors = generate_random_colors(2)
    
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
    
    # Find initial features
    context.p0 = cv2.goodFeaturesToTrack(context.old_gray, mask=binary_mask, **context.feature_params, maxCorners=context.max_point)
    
    if context.p0 is None:
        return False
    
    # Initialize mask
    context.mask = np.zeros_like(frame)
    
    # Create tracked points with unique IDs
    context.tracked_points.clear()
    context.point_paths.clear()
    context.point_clusters.clear()
    
    for point in context.p0.reshape(-1, 2):
        tracked_pt = TrackedPoint(point)
        context.tracked_points[tracked_pt.id] = tracked_pt
        context.point_paths[tracked_pt.id] = []
        context.point_clusters[tracked_pt.id] = 0
    
    return True

def process_frame(frame, context):
    """Process a single frame with optical flow"""
    if frame is None:
        return None, None, None
    
    # Check if frame is gray or color
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame_gray = frame.copy()
    else:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
    # Initialize tracking if needed
    if context.p0 is None or context.old_gray is None:
        if not initialize_tracking(frame_gray, context):
            return None, None, None
        return None, None, None
        
    # Get current tracked point IDs and positions
    point_ids = list(context.tracked_points.keys())
    if len(point_ids) == 0:
        initialize_tracking(frame_gray, context)
        return None, None, None
    
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
        return None, None, None
    
    uvs = (good_new - good_old)

    # Update paths and positions for each tracked point
    for i, pid in enumerate(good_ids):
        # Update position
        context.tracked_points[pid].position = good_new[i]
        
        # Update path
        context.point_paths[pid].append(good_new[i])
        # Keep only recent path history
        if len(context.point_paths[pid]) > context.max_path_length:
            context.point_paths[pid] = context.point_paths[pid][-context.max_path_length:]

    # Update for next iteration
    context.old_gray = frame_gray.copy()
    context.p0 = good_new.reshape(-1, 1, 2)

    # Reprocess good features after certain number of frames
    if context.frame_iter >= int(0.5 * context.fps) - 1 and len(good_new) < context.max_point:
        # Get current good points as starting point
        tmp_p = good_new.copy()
        
        # Create binary mask for goodFeaturesToTrack (only 0 and 255)
        binary_mask = (context.detection_mask > 200).astype(np.uint8) * 255

        #Create a mask to avoid adding new points too close to existing ones
        for existing_pt in good_new:
            x, y = int(existing_pt[0]), int(existing_pt[1])
            cv2.circle(binary_mask, (x, y), 4, 0, -1)  # Block a radius of 4 pixels around existing points
        
        # Find new feature points using detection mask
        # print(f"Finding new features. Current points: {len(good_new)}, Max points: {context.max_point}, New maxCorners: {context.max_point - len(good_new)}")
        new_features = cv2.goodFeaturesToTrack(context.old_gray, mask=binary_mask, **context.feature_params, maxCorners=context.max_point - len(good_new))
        
        if new_features is not None:
            new_features = new_features.reshape(-1, 2)
            
            # Add new features that are not too close to existing ones
            for new_pt in new_features:
                
                # Create new tracked point with unique ID
                tracked_pt = TrackedPoint(new_pt)
                context.tracked_points[tracked_pt.id] = tracked_pt
                context.point_paths[tracked_pt.id] = []
                context.point_clusters[tracked_pt.id] = 0
                # Add to p0 for next iteration
                tmp_p = np.vstack([tmp_p, new_pt]) if len(tmp_p) > 0 else new_pt.reshape(-1, 2)
            
            # Update p0 for next iteration
            context.p0 = tmp_p.reshape(-1, 1, 2)
        
        # Reset iteration counter and clear tracking data
        context.frame_iter = -1

    context.mask = np.zeros_like(frame)  # Reset mask
    context.frame_iter += 1

    # print(f"Tracked points: {len(context.tracked_points)}")
    
    return good_new, good_ids, uvs

def cleanup(context):
    """Cleanup optical flow resources"""
    if context:
        context.p0 = None
        context.old_gray = None
        context.mask = None
        context.paths = []
        context.tracked_points.clear()
        context.point_paths.clear()
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
    context = setup(50)
    
    print(f"Processing video: {filename}")
    print(f"FPS: {fps}")

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (853, 480))
    
        # Process frame
        good_new, good_ids, uvs = process_frame(frame, context)
  

        # draw uvs on the good_new points
        for i, pid in enumerate(good_ids) if good_new is not None else []:
            new = good_new[i]
            a, b = int(new[0]), int(new[1])
            u, v = uvs[i] * fps

            # Draw arrow for optical flow
            frame = cv2.circle(frame, (a, b), 5, context.colors[0], -1)
            frame = cv2.arrowedLine(frame, (a, b), (int(a + u), int(b + v)), context.colors[1], 2, tipLength=0.2)
        
        # Display result
        cv2.imshow("Optical Flow", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
    
    # Cleanup
    capture.release()
    cv2.destroyAllWindows()
    cleanup(context)