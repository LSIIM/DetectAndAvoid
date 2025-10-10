#!/usr/bin/env python3
"""
DetectAndAvoid - Main Integration Module

This module integrates YOLO detection, Sky Segmentation, and Optical Flow
processing into a unified video processing pipeline.

Usage:
    python main.py <video_path> [--clusters <num>] [--confidence <conf>]
"""

import cv2
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Add module paths to sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "Yolo" / "Yolo11"))
sys.path.append(str(PROJECT_ROOT / "Sky_Seg"))
sys.path.append(str(PROJECT_ROOT / "OpticalFlow"))

# Import modules (assuming they are modularized)
try:
    import yolo_detector
    import sky_segmentation
    import optical_flow
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all modules are properly modularized")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DetectAndAvoid Integrated Processing")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters for optical flow (default: 5)")
    parser.add_argument("--confidence", type=float, default=0.6, help="YOLO confidence threshold (default: 0.6)")
    parser.add_argument("--output", help="Output video path (optional)")
    parser.add_argument("--resize-height", type=int, default=480, help="Resize frame height (default: 480)")
    
    return parser.parse_args()


def setup_video_capture(video_path):
    """Setup video capture and get properties"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return cap, fps, frame_width, frame_height


def setup_video_writer(output_path, fps, width, height):
    """Setup video writer if output path is provided"""
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return None


def main():
    """Main integration function"""
    args = parse_arguments()
    
    print("=== DetectAndAvoid Integration System ===")
    print(f"Video: {args.video_path}")
    print(f"Clusters: {args.clusters}")
    print(f"YOLO Confidence: {args.confidence}")
    print("==========================================")
    
    # Setup video capture
    try:
        cap, fps, orig_width, orig_height = setup_video_capture(args.video_path)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Calculate resize scale
    resize_scale = args.resize_height / orig_height
    processing_width = int(orig_width * resize_scale)
    processing_height = args.resize_height
    
    print(f"Original resolution: {orig_width}x{orig_height}")
    print(f"Processing resolution: {processing_width}x{processing_height}")
    print(f"FPS: {fps}")
    
    # Setup video writer
    writer = setup_video_writer(args.output, fps, processing_width * 3, processing_height)
    
    # Setup modules
    print("\n--- Setting up modules ---")
    
    try:
        # YOLO setup
        print("Setting up YOLO detector...")
        yolo_context = yolo_detector.setup(confidence=args.confidence)
        
        # Sky Segmentation setup
        print("Setting up Sky Segmentation...")
        sky_context = sky_segmentation.setup()
        
        # Optical Flow setup
        print("Setting up Optical Flow...")
        flow_context = optical_flow.setup(
            clusters=args.clusters, 
            fps=fps,
            processing_size=(processing_width, processing_height)
        )
        
        print("All modules setup successfully!")
        
    except Exception as e:
        print(f"Error setting up modules: {e}")
        cap.release()
        if writer:
            writer.release()
        return 1
    
    # Main processing loop
    print("\n--- Starting video processing ---")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Resize frame for processing
            resized_frame = cv2.resize(frame, (processing_width, processing_height))
            
            # Process frame with each module
            yolo_result = yolo_detector.process_frame(resized_frame, yolo_context)
            sky_result = sky_segmentation.process_frame(resized_frame, sky_context)
            flow_result = optical_flow.process_frame(resized_frame, flow_context)
            
            # Create combined display
            combined_frame = np.hstack([yolo_result, sky_result, flow_result])
            
            # Add frame info
            info_text = f"Frame: {frame_count} | YOLO | Sky Seg | Optical Flow"
            cv2.putText(combined_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Write frame if output is specified
            if writer:
                writer.write(combined_frame)
            
            # Display results
            cv2.imshow("DetectAndAvoid - YOLO | Sky Segmentation | Optical Flow", combined_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('q'):
                break
            elif key == ord('s'):  # Save current frame
                cv2.imwrite(f"frame_{frame_count:06d}.jpg", combined_frame)
                print(f"Saved frame {frame_count}")
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        # Cleanup
        print(f"\n--- Processing completed ---")
        print(f"Total frames processed: {frame_count}")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Cleanup modules
        try:
            yolo_detector.cleanup(yolo_context)
            sky_segmentation.cleanup(sky_context)
            optical_flow.cleanup(flow_context)
        except:
            pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())