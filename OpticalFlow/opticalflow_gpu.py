import cv2
import numpy as np
import time
import skfuzzy as fuzz
import random
import sys

class OpticalFlowGPUContext:
    """Context class to store GPU-accelerated optical flow state"""
    def __init__(self):
        self.number_clusters = 5
        self.fps = 30
        self.processing_size = (640, 480)
        self.max_point = 100
        self.max_path_length = 10
        
        # Check CUDA availability
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if not self.cuda_available:
            print("WARNING: CUDA not available, falling back to CPU processing")
        else:
            print(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=self.max_point,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Lucas-Kanade parameters for GPU
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Create GPU optical flow object if CUDA available
        if self.cuda_available:
            self.gpu_sparse_flow = cv2.cuda.SparsePyrLKOpticalFlow_create(
                winSize=self.lk_params['winSize'],
                maxLevel=self.lk_params['maxLevel'],
                iters=10,
                useInitialFlow=False
            )
            
            # Create GPU corner detector
            self.gpu_corner_detector = cv2.cuda.createGoodFeaturesToTrackDetector(
                srcType=cv2.CV_8UC1,
                maxCorners=self.max_point,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
        
        # State variables
        self.p0 = None
        self.old_gray = None
        self.old_gray_gpu = None  # GPU version
        self.mask = None
        self.mask_gpu = None  # GPU version
        self.paths = []
        self.frame_iter = 0
        self.previous_centroids = None
        self.cluster_id_mapping = None
        self.colors = None
        self.start_time = None
        
        # GPU streams for async operations
        if self.cuda_available:
            self.stream = cv2.cuda.Stream()

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
    """Setup GPU-accelerated optical flow processing"""
    context = OpticalFlowGPUContext()
    context.number_clusters = clusters
    context.fps = fps
    context.processing_size = processing_size
    
    # Generate colors for clusters
    context.colors = generate_random_colors(max(clusters, 10))
    
    context.start_time = time.perf_counter()
    
    return context

def initialize_tracking(frame, context):
    """Initialize tracking on first frame using GPU"""
    if frame is None:
        return False
    
    # Resize frame using GPU if available
    if context.cuda_available:
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Resize on GPU
        gpu_resized = cv2.cuda.resize(gpu_frame, context.processing_size)
        
        # Convert to grayscale on GPU
        context.old_gray_gpu = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2GRAY)
        
        # Download to CPU for initial processing
        context.old_gray = context.old_gray_gpu.download()
        
        # Detect features on GPU
        gpu_corners = context.gpu_corner_detector.detect(context.old_gray_gpu)
        
        if gpu_corners.size().height > 0:
            context.p0 = gpu_corners.download().reshape(-1, 1, 2)
        else:
            context.p0 = None
        
        # Initialize mask
        frame_resized = gpu_resized.download()
        context.mask = np.zeros_like(frame_resized)
        context.mask_gpu = cv2.cuda_GpuMat()
        context.mask_gpu.upload(context.mask)
    else:
        # CPU fallback
        frame_resized = cv2.resize(frame, context.processing_size)
        context.old_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        context.p0 = cv2.goodFeaturesToTrack(context.old_gray, mask=None, **context.feature_params)
        context.mask = np.zeros_like(frame_resized)
    
    if context.p0 is None:
        return False
    
    context.paths = []
    return True

def process_frame(frame, context):
    """Process a single frame with GPU-accelerated optical flow"""
    if frame is None:
        return frame
    
    # Initialize tracking if needed
    if context.p0 is None or context.old_gray is None:
        if not initialize_tracking(frame, context):
            if context.cuda_available:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_resized = cv2.cuda.resize(gpu_frame, context.processing_size)
                return gpu_resized.download()
            return cv2.resize(frame, context.processing_size)
        if context.cuda_available:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_resized = cv2.cuda.resize(gpu_frame, context.processing_size)
            return gpu_resized.download()
        return cv2.resize(frame, context.processing_size)
    
    if context.cuda_available:
        # GPU processing
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Resize on GPU
        gpu_resized = cv2.cuda.resize(gpu_frame, context.processing_size, stream=context.stream)
        
        # Convert to grayscale on GPU
        gpu_gray = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2GRAY, stream=context.stream)
        
        # Upload points to GPU
        gpu_p0 = cv2.cuda_GpuMat()
        gpu_p0.upload(context.p0)
        
        # Calculate optical flow on GPU
        gpu_p1, gpu_status = context.gpu_sparse_flow.calc(
            context.old_gray_gpu, gpu_gray, gpu_p0, None, stream=context.stream
        )
        
        # Synchronize stream
        context.stream.waitForCompletion()
        
        # Download results
        p1 = gpu_p1.download()
        st = gpu_status.download()
        frame_resized = gpu_resized.download()
        
        # Update GPU state
        context.old_gray_gpu = gpu_gray
    else:
        # CPU fallback
        frame_resized = cv2.resize(frame, context.processing_size)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(context.old_gray, frame_gray, context.p0, None, **context.lk_params)
        context.old_gray = frame_gray.copy()
    
    # Select good points (CPU processing)
    good_new = p1[st.flatten() == 1]
    good_old = context.p0[st.flatten() == 1]
    
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
            context.paths = context.paths[:len(good_new)]
        else:
            context.paths.extend([[] for _ in range(len(good_new) - len(context.paths))])
    
    # Update each point's path with current velocity
    for i in range(len(good_new)):
        if i < len(uvs):
            context.paths[i].append(uvs[i])
            if len(context.paths[i]) > context.max_path_length:
                context.paths[i] = context.paths[i][-context.max_path_length:]
    
    # Calculate average path vectors for clustering
    path_vectors = []
    for path in context.paths:
        if len(path) > 0:
            avg_path = np.mean(path, axis=0)
            path_vectors.append(avg_path)
        else:
            path_vectors.append(np.array([0.0, 0.0]))
    
    path_vectors = np.array(path_vectors)

    # Perform fuzzy clustering if we have enough points
    cluster_membership = []
    if len(good_new) >= context.number_clusters:
        alldata = np.vstack((good_new[:, 0], good_new[:, 1], uvs[:, 0], uvs[:, 1], path_vectors[:, 0], path_vectors[:, 1]))
        
        try:
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                alldata, context.number_clusters, 2, error=0.005, maxiter=50, init=None)
            
            raw_membership = np.argmax(u, axis=0)
            
            # Handle cluster tracking for consistent colors
            if context.previous_centroids is not None:
                distances = np.zeros((len(cntr), len(context.previous_centroids)))
                for i, curr_center in enumerate(cntr):
                    for j, prev_center in enumerate(context.previous_centroids):
                        pos_dist = np.linalg.norm(curr_center[:2] - prev_center[:2])
                        vel_dist = np.linalg.norm(curr_center[2:4] - prev_center[2:4])
                        path_dist = np.linalg.norm(curr_center[4:6] - prev_center[4:6])
                        distances[i, j] = 0.5 * pos_dist + 0.25 * vel_dist + 0.25 * path_dist
                
                used_prev_ids = set()
                new_mapping = {}
                
                flat_indices = np.unravel_index(np.argsort(distances, axis=None), distances.shape)
                for curr_id, prev_id in zip(flat_indices[0], flat_indices[1]):
                    if curr_id not in new_mapping and prev_id not in used_prev_ids:
                        new_mapping[curr_id] = context.cluster_id_mapping[prev_id] if context.cluster_id_mapping is not None else prev_id
                        used_prev_ids.add(prev_id)
                
                available_ids = set(range(context.number_clusters)) - set(new_mapping.values())
                for curr_id in range(len(cntr)):
                    if curr_id not in new_mapping:
                        if available_ids:
                            new_mapping[curr_id] = available_ids.pop()
                        else:
                            new_mapping[curr_id] = curr_id
                
                context.cluster_id_mapping = new_mapping
                cluster_membership = [context.cluster_id_mapping[raw_membership[i]] for i in range(len(raw_membership))]
            else:
                context.cluster_id_mapping = {i: i for i in range(context.number_clusters)}
                cluster_membership = raw_membership.tolist()
            
            context.previous_centroids = cntr.copy()
            
        except:
            cluster_membership = [0] * len(good_new)
    else:
        cluster_membership = [0] * len(good_new)

    # Draw the tracks with cluster colors
    result_frame = frame_resized.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = int(new[0]), int(new[1])
        c, d = int(old[0]), int(old[1])
        
        cluster_id = cluster_membership[i] if i < len(cluster_membership) else 0
        color = context.colors[cluster_id % len(context.colors)]
        
        context.mask = cv2.line(context.mask, (a, b), (c, d), color, 2)
        result_frame = cv2.circle(result_frame, (a, b), 5, color, -1)

    # Draw cluster centroids and their velocity vectors
    if len(good_new) >= context.number_clusters and 'cntr' in locals():
        for i, centroid in enumerate(cntr):
            if i < len(context.colors):
                mapped_id = context.cluster_id_mapping.get(i, i) if context.cluster_id_mapping else i
                color = context.colors[mapped_id % len(context.colors)]
                
                center_pos = (int(centroid[0]), int(centroid[1]))
                vel_magnitude = np.sqrt(centroid[2]**2 + centroid[3]**2)
                vel_text = f"V:{vel_magnitude:.1f}"
                
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
    
    # Display FPS on frame with GPU indicator
    gpu_indicator = " [GPU]" if context.cuda_available else " [CPU]"
    fps_text = f"FPS: {int(processing_fps)}{gpu_indicator}"
    cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Update for next iteration
    context.p0 = good_new.reshape(-1, 1, 2)

    # Reprocess good features after certain number of frames
    if context.frame_iter >= int(0.5 * context.fps) - 1:
        tmp_p = good_new.copy()
        
        if context.cuda_available:
            # Detect new features on GPU
            gpu_corners = context.gpu_corner_detector.detect(context.old_gray_gpu)
            if gpu_corners.size().height > 0:
                new_features = gpu_corners.download().reshape(-1, 2)
            else:
                new_features = None
        else:
            # CPU fallback
            new_features_result = cv2.goodFeaturesToTrack(context.old_gray, mask=None, **context.feature_params)
            new_features = new_features_result.reshape(-1, 2) if new_features_result is not None else None
        
        if new_features is not None:
            # Add new features that are not too close to existing ones
            for new_pt in new_features:
                add_point = True
                for existing_pt in good_new:
                    if np.sum((existing_pt - new_pt)**2) < 16:
                        add_point = False
                        break
                
                if add_point:
                    tmp_p = np.vstack([tmp_p, new_pt]) if len(tmp_p) > 0 else new_pt.reshape(-1, 2)
            
            context.p0 = tmp_p.reshape(-1, 1, 2)
        
        # Reset iteration counter and clear tracking data
        context.frame_iter = -1
        context.paths = []
        context.mask = np.zeros_like(frame_resized)
        if context.cuda_available:
            context.mask_gpu.upload(context.mask)
    
    context.frame_iter += 1
    context.start_time = end_time
    
    return img

def cleanup(context):
    """Cleanup optical flow resources"""
    if context:
        context.p0 = None
        context.old_gray = None
        context.old_gray_gpu = None
        context.mask = None
        context.mask_gpu = None
        context.paths = []
        if context.cuda_available:
            context.stream = None

# Standalone execution for testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python opticalflow_gpu.py <clusters> [video_path]")
        print("  <clusters>: Number of clusters (required)")
        print("  [video_path]: Path to video file (optional)")
        sys.exit(1)

    number_clusters = int(sys.argv[1])

    if len(sys.argv) >= 3:
        filename = sys.argv[2]
    else:
        filename = "C:\\Users\\dudu1\\Desktop\\DeA\\DetectAndAvoid\\Videos\\fev_corte_2.mp4"

    capture = cv2.VideoCapture(filename)
    if not capture.isOpened():
        print(f"Error: Could not open video {filename}")
        sys.exit(1)

    fps = capture.get(cv2.CAP_PROP_FPS)
    
    context = setup(clusters=number_clusters, fps=fps)
    
    print(f"Processing video: {filename}")
    print(f"Clusters: {number_clusters}")
    print(f"FPS: {fps}")
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        result = process_frame(frame, context)
        
        cv2.imshow("Optical Flow GPU", result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            if context.number_clusters < 10:
                context.number_clusters += 1
                context.previous_centroids = None
                context.cluster_id_mapping = None
                print(f"Clusters increased to: {context.number_clusters}")
        elif key == ord('-'):
            if context.number_clusters > 2:
                context.number_clusters -= 1
                context.previous_centroids = None
                context.cluster_id_mapping = None
                print(f"Clusters decreased to: {context.number_clusters}")

    capture.release()
    cv2.destroyAllWindows()
    cleanup(context)
