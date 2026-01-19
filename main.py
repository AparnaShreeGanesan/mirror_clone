import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import RunningMode, HandLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
import numpy as np
import math
import random
import time

class MirrorCloneFX:
    def __init__(self):
        # Initialize MediaPipe HandLandmarker
        base_options = BaseOptions(model_asset_path='hand_landmarker.task')
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_hands=1
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        
        # Visual modes
        self.modes = {
            0: "Dots",
            1: "Lines", 
            2: "ASCII",
            3: "Particles"
        }
        self.current_mode = 0
        
        # Particles system
        self.particles = []
        self.max_particles = 200
        
        # ASCII characters for ASCII mode (from dense to sparse)
        self.ascii_chars = "█▉▊▋▌▍▎▏ "
        
        # Window dimensions
        self.window_width = 1280
        self.window_height = 720
        self.half_width = self.window_width // 2
        
    def detect_hand_gesture(self, landmarks):
        """Detect hand gestures from landmarks"""
        if not landmarks or len(landmarks) == 0:
            return None
            
        # Get landmark positions (landmarks is a NormalizedLandmarkList)
        hand_landmarks = landmarks[0] if isinstance(landmarks, (list, tuple)) else landmarks
        
        # Convert to list format for easier access
        landmarks_list = list(hand_landmarks) if hasattr(landmarks, '__iter__') else []
        
        if len(landmarks_list) < 21:
            return None
            
        # Get specific landmark positions
        thumb_tip = landmarks_list[4]
        thumb_ip = landmarks_list[3]
        index_tip = landmarks_list[8]
        index_pip = landmarks_list[6]
        middle_tip = landmarks_list[12]
        middle_pip = landmarks_list[10]
        ring_tip = landmarks_list[16]
        ring_pip = landmarks_list[14]
        pinky_tip = landmarks_list[20]
        pinky_pip = landmarks_list[18]
        
        # Check if fingers are extended
        fingers_up = []
        
        # Thumb (different logic - compare x coordinates)
        if thumb_tip.x > thumb_ip.x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
            
        # Other fingers (compare y coordinates)
        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        finger_pips = [index_pip, middle_pip, ring_pip, pinky_pip]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if tip.y < pip.y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # Gesture recognition
        # One finger (index) - Lines mode
        if fingers_up == [0, 1, 0, 0, 0]:
            return 1
        
        # Two fingers (index + middle) - Dots mode  
        elif fingers_up == [0, 1, 1, 0, 0]:
            return 0
            
        # Thumb + pinky - ASCII mode
        elif fingers_up == [1, 0, 0, 0, 1]:
            return 2
            
        # Open palm (all fingers) - Particles mode
        elif sum(fingers_up) >= 4:
            return 3
            
        return None
    
    def create_dots_effect(self, frame):
        """Create stippled dot rendering"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.zeros_like(frame)
        
        height, width = gray.shape
        dot_spacing = 12  # Increased spacing for cleaner look
        
        for y in range(0, height, dot_spacing):
            for x in range(0, width, dot_spacing):
                if y < height and x < width:
                    intensity = gray[y, x]
                    if intensity > 60:  # Adjusted threshold
                        # Variable dot size based on intensity
                        radius = int((intensity / 255) * 6) + 1
                        # Use original color but make it more vibrant
                        color = frame[y, x].astype(int)
                        # Enhance colors slightly
                        color = np.clip(color * 1.2, 0, 255).astype(int)
                        cv2.circle(result, (x, y), radius, color.tolist(), -1)
        
        return result
    
    def create_lines_effect(self, frame):
        """Create edge outline rendering"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with better parameters
        edges = cv2.Canny(blurred, 30, 80)
        
        # Create colored edge image with original colors
        result = np.zeros_like(frame)
        
        # Use original frame colors for edges
        edge_points = np.where(edges > 0)
        for y, x in zip(edge_points[0], edge_points[1]):
            # Get original color and make it brighter
            original_color = frame[y, x].astype(int)
            enhanced_color = np.clip(original_color * 1.5, 0, 255).astype(int)
            result[y, x] = enhanced_color
        
        # Add some glow effect
        kernel = np.ones((3,3), np.uint8)
        result = cv2.dilate(result, kernel, iterations=1)
        
        return result
    
    def create_ascii_effect(self, frame):
        """Create ASCII art rendering"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.zeros_like(frame)
        
        height, width = gray.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Increased spacing for better readability
        char_width = 16
        char_height = 20
        
        for y in range(0, height, char_height):
            for x in range(0, width, char_width):
                if y + char_height < height and x + char_width < width:
                    # Get average intensity in this region
                    region = gray[y:y+char_height, x:x+char_width]
                    avg_intensity = np.mean(region)
                    
                    # Only draw characters for areas with sufficient contrast
                    if avg_intensity > 30:  # Skip very dark areas
                        # Map intensity to ASCII character (inverted for better contrast)
                        char_index = int(((255 - avg_intensity) / 255) * (len(self.ascii_chars) - 1))
                        char = self.ascii_chars[char_index]
                        
                        # Use a more visible color scheme
                        if avg_intensity > 150:
                            color = [255, 255, 255]  # White for bright areas
                        elif avg_intensity > 100:
                            color = [0, 255, 0]      # Green for medium areas
                        else:
                            color = [0, 255, 255]    # Cyan for darker areas
                        
                        # Draw character with better positioning
                        cv2.putText(result, char, (x + 2, y + char_height - 4), 
                                   font, font_scale, color, thickness)
        
        return result
    
    def update_particles(self, frame, landmarks):
        """Update particle system"""
        if landmarks and len(landmarks) > 0:
            # Get the first hand's landmarks
            hand_landmarks = landmarks[0] if isinstance(landmarks, (list, tuple)) else landmarks
            landmarks_list = list(hand_landmarks) if hasattr(hand_landmarks, '__iter__') else []
            
            # Add new particles near hand landmarks
            for landmark in landmarks_list[::2]:  # Every other landmark to reduce particles
                if len(self.particles) < self.max_particles:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    
                    particle = {
                        'x': x + random.randint(-20, 20),
                        'y': y + random.randint(-20, 20),
                        'vx': random.uniform(-2, 2),
                        'vy': random.uniform(-2, 2),
                        'life': 60,
                        'color': [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]
                    }
                    self.particles.append(particle)
        
        # Update existing particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        for particle in self.particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            particle['vy'] += 0.1  # Gravity
    
    def create_particles_effect(self, frame, landmarks):
        """Create particle effect"""
        result = np.zeros_like(frame)
        
        self.update_particles(frame, landmarks)
        
        # Draw particles
        for particle in self.particles:
            if 0 <= particle['x'] < frame.shape[1] and 0 <= particle['y'] < frame.shape[0]:
                alpha = particle['life'] / 60.0
                radius = max(1, int(alpha * 4))
                color = [int(c * alpha) for c in particle['color']]
                cv2.circle(result, (int(particle['x']), int(particle['y'])), 
                          radius, color, -1)
        
        return result
    
    def process_frame(self, frame, landmarks):
        """Process frame based on current mode"""
        if self.current_mode == 0:  # Dots
            return self.create_dots_effect(frame)
        elif self.current_mode == 1:  # Lines
            return self.create_lines_effect(frame)
        elif self.current_mode == 2:  # ASCII
            return self.create_ascii_effect(frame)
        elif self.current_mode == 3:  # Particles
            return self.create_particles_effect(frame, landmarks)
        else:
            return frame
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("MirrorCloneFX started!")
        print("Hand gestures:")
        print("✌️  Two fingers → Dots mode")
        print("☝️  One finger → Lines mode") 
        print("🤙 Thumb + pinky → ASCII mode")
        print("✋ Open palm → Particles mode")
        print("Press 'q' to quit")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Process hands with timestamp for video mode
            results = self.detector.detect_for_video(mp_image, int(frame_count * 33.33))  # ~30fps
            landmarks = None
            
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                landmarks = results.hand_landmarks
                
                # Detect gesture and update mode
                gesture = self.detect_hand_gesture(landmarks)
                if gesture is not None:
                    self.current_mode = gesture
            
            # Resize frame to half width for split view
            frame_resized = cv2.resize(frame, (self.half_width, self.window_height))
            
            # Create stylized version
            stylized = self.process_frame(frame, landmarks)
            stylized_resized = cv2.resize(stylized, (self.half_width, self.window_height))
            
            # Create split screen
            split_screen = np.hstack([frame_resized, stylized_resized])
            
            # Add mode indicator
            mode_text = f"Mode: {self.modes[self.current_mode]}"
            cv2.putText(split_screen, mode_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add divider line
            cv2.line(split_screen, (self.half_width, 0), 
                    (self.half_width, self.window_height), (255, 255, 255), 2)
            
            # Add labels
            cv2.putText(split_screen, "Original", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(split_screen, "Clone", (self.half_width + 10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display
            cv2.imshow('MirrorCloneFX', split_screen)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    app = MirrorCloneFX()
    app.run()

if __name__ == "__main__":
    main()