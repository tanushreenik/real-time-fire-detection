import cv2
import numpy as np
import os
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


class VideoFireDetector:
    def __init__(
        self,
        model_path="fire_detection_model.h5",
        video_source=0,
        confidence_threshold=0.5,
        alert_frames=10,
    ):
        """
        Initialize the video fire detection system.

        Args:
            model_path: Path to the trained fire detection model
            video_source: Camera index or video file path
            confidence_threshold: Minimum confidence to consider as fire detection
            alert_frames: Number of consecutive frames with detection before alerting
        """
        self.model_path = model_path
        self.video_source = video_source
        self.confidence_threshold = confidence_threshold
        self.alert_frames = alert_frames
        self.model = None

        # Detection tracking
        self.detection_counter = 0
        self.alert_active = False

        # Create output directory for saved frames
        self.output_dir = "detected_events"
        os.makedirs(self.output_dir, exist_ok=True)

        # Load the model
        self.load_detection_model()

    def load_detection_model(self):
        """Load the trained fire detection model"""
        try:
            # self.model = load_model(self.model_path)
            # Load without compiling to avoid the 'lr' error
            self.model = load_model(self.model_path, compile=False)
            # Then compile manually with the correct arguments
            optimizer = Adam(learning_rate=1e-5)  # Or any value you need
            self.model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using a dummy model for demonstration")
            # Create a simple dummy model for demonstration
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
            from tensorflow.keras.applications import MobileNetV2

            base_model = MobileNetV2(
                weights="imagenet", include_top=False, input_shape=(224, 224, 3)
            )
            self.model = Sequential(
                [base_model, GlobalAveragePooling2D(), Dense(2, activation="softmax")]
            )
            print("Dummy model created")

    def prepare_frame(self, frame):
        """
        Prepare a video frame for prediction.

        Args:
            frame: Input video frame

        Returns:
            Preprocessed frame ready for model prediction
        """
        # Resize frame to match model input size
        resized = cv2.resize(frame, (224, 224))

        # Convert to array and preprocess
        img_array = img_to_array(resized)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(img_array)

        return preprocessed_img

    def detect_fire_in_frame(self, frame):
        """
        Detect fire in a single video frame.

        Args:
            frame: Input video frame

        Returns:
            Tuple of (is_fire_detected, fire_confidence, result_frame)
        """
        # Prepare frame
        preprocessed_frame = self.prepare_frame(frame)

        # Make prediction
        predictions = self.model.predict(preprocessed_frame, verbose=0)

        # Get fire confidence score (assuming first class is "fire")
        fire_confidence = predictions[0][0]
        no_fire_confidence = predictions[0][1]

        # Determine if fire is detected
        is_fire_detected = fire_confidence > self.confidence_threshold

        # Create visualization
        result_frame = frame.copy()

        # Add overlay and text based on detection result
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Always display the confidence scores
        confidence_text = f"Fire: {fire_confidence*100:.2f}% | Not Fire: {no_fire_confidence*100:.2f}%"
        cv2.putText(
            result_frame, confidence_text, (10, 30), font, 0.7, (255, 255, 255), 2
        )

        if is_fire_detected:
            # Add red rectangle around the frame
            cv2.rectangle(
                result_frame,
                (0, 0),
                (result_frame.shape[1], result_frame.shape[0]),
                (0, 0, 255),
                10,
            )

            # Add "FIRE DETECTED" text
            cv2.putText(
                result_frame, "FIRE DETECTED", (10, 70), font, 1, (0, 0, 255), 3
            )

        if self.alert_active:
            # Add flashing "ALERT!" text when alert is active
            if int(time.time() * 2) % 2 == 0:  # Flash twice per second
                cv2.putText(
                    result_frame,
                    "ALERT!",
                    (result_frame.shape[1] // 2 - 100, result_frame.shape[0] // 2),
                    font,
                    2,
                    (0, 0, 255),
                    5,
                )

        return is_fire_detected, fire_confidence, result_frame

    def save_fire_frame(self, frame):
        """
        Save a frame when fire is detected.

        Args:
            frame: Frame to save
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{self.output_dir}/fire_detected_{timestamp}.jpg"

        # Save the frame
        cv2.imwrite(save_path, frame)
        print(f"Fire detection saved to {save_path}")

    def sound_alarm(self):
        """
        Trigger an alarm when fire is detected.
        In a real system, this would connect to an actual alarm system.
        """
        print("\a")  # ASCII bell character - makes a beep sound on many systems
        print("🔥 ALERT! Fire detected! 🔥")

        # In a real system, you might send an email, SMS, or trigger a physical alarm

    def process_video(self):
        """Main loop for processing video and detecting fire"""
        # Open video capture
        cap = cv2.VideoCapture(self.video_source)

        if not cap.isOpened():
            print(f"Error: Couldn't open video source {self.video_source}")
            return

        print(f"Starting fire detection on source: {self.video_source}")
        print(f"Using model: {self.model_path}")
        print("Press 'q' to quit")

        # For FPS calculation
        frame_count = 0
        start_time = time.time()
        fps = 0

        while True:
            # Read frame
            ret, frame = cap.read()

            if not ret:
                print("End of video stream")
                break

            # Update frame count for FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Detect fire in frame
            is_fire_detected, confidence, result_frame = self.detect_fire_in_frame(
                frame
            )

            # Update detection counter
            if is_fire_detected:
                self.detection_counter += 1
            else:
                self.detection_counter = 0

            # Check if we should trigger an alert
            if self.detection_counter >= self.alert_frames and not self.alert_active:
                self.alert_active = True
                self.sound_alarm()
                self.save_fire_frame(result_frame)

            # Reset alert if no detections
            if self.detection_counter == 0:
                self.alert_active = False

            # Display FPS
            cv2.putText(
                result_frame,
                f"FPS: {fps:.2f}",
                (result_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Display frame number when fire is detected (for debugging)
            if is_fire_detected:
                cv2.putText(
                    result_frame,
                    f"Frame count: {self.detection_counter}/{self.alert_frames}",
                    (10, result_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            # Show the frame
            cv2.imshow("Fire Detection", result_frame)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()


# Run the detector
if __name__ == "__main__":
    # You can use camera index (usually 0 for webcam) or video file path
    detector = VideoFireDetector(
        model_path="saved_model/raks_model14.h5",  # Path to your model
        video_source=0,  # 0 for webcam, or path to video file
        confidence_threshold=0.5,  # Minimum confidence for fire detection
        alert_frames=10,  # Number of consecutive frames before alerting
    )
    detector.process_video()
