"""GUI for directional people counter"""

import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
import threading
import cv2
from PIL import Image, ImageTk

from utils.person_tracker import PersonTracker
from utils.helpers import draw_detections
from counter import DirectionalCounter


class CounterGUI:
    """
    Provides interface for:
    - Video file selection
    - Counting line configuration
    - Real-time video processing and visualization
    - Live statistics display
    - Ground truth comparison
    """
    
    def __init__(self, root):
        """
        Initialize the GUI application.
        
        Args:
            root: Tkinter root window.
        """

        self.root = root
        self.root.title("Directional People Counter")
        self.root.geometry("1400x800")
        
        self.video_path = None
        self.is_playing = False
        self.tracker = None
        self.counter = None
        self.cap = None
        self.last_stats_str = ""
        
        self.line_position = tk.DoubleVar(value=0.5)
        self.orientation = tk.StringVar(value="vertical")
        self.count_direction = tk.StringVar(value="right")
        self.min_track_age = tk.IntVar(value=5)
        
        self.setup_ui()
    
    def setup_ui(self):
        """
        Create and configure the GUI layout.
        
        Setup:
        - Control panel
        - Video display
        - Statistics panel
        - Status bar
        """
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights (window resizing)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))
        
        # File selection
        ttk.Button(control_frame, text="Select Video", command=self.select_video).grid(
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5
        )
        
        self.video_label = ttk.Label(control_frame, text="No video selected", wraplength=200)
        self.video_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )
        
        # Line orientation
        ttk.Label(control_frame, text="Line Orientation:").grid(row=3, column=0, sticky=tk.W, pady=5)
        orientation_frame = ttk.Frame(control_frame)
        orientation_frame.grid(row=4, column=0, columnspan=2, pady=5)
        
        ttk.Radiobutton(
            orientation_frame, text="Horizontal", variable=self.orientation, 
            value="horizontal", command=self.update_direction_options
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            orientation_frame, text="Vertical", variable=self.orientation, 
            value="vertical", command=self.update_direction_options
        ).pack(side=tk.LEFT, padx=5)
        
        # Count direction
        ttk.Label(control_frame, text="Count Direction:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.direction_menu = ttk.Combobox(
            control_frame, textvariable=self.count_direction, 
            values=["up", "down"], state="readonly", width=15
        )
        self.direction_menu.grid(row=6, column=0, columnspan=2, pady=5)
        
        # Line position
        ttk.Label(control_frame, text="Line Position:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.position_label = ttk.Label(control_frame, text="0.50")
        self.position_label.grid(row=7, column=1, sticky=tk.E, pady=5)
        
        position_slider = ttk.Scale(
            control_frame, from_=0.1, to=0.9, variable=self.line_position,
            orient=tk.HORIZONTAL, command=self.update_position_label
        )
        position_slider.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Min track age
        ttk.Label(control_frame, text="Min Track Age:").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.age_label = ttk.Label(control_frame, text="5")
        self.age_label.grid(row=9, column=1, sticky=tk.E, pady=5)
        
        age_slider = ttk.Scale(
            control_frame, from_=1, to=20, variable=self.min_track_age,
            orient=tk.HORIZONTAL, command=self.update_age_label
        )
        age_slider.grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )
        
        # Control buttons
        self.start_button = ttk.Button(
            control_frame, text="Start", command=self.start_processing, state=tk.DISABLED
        )
        self.start_button.grid(row=12, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.stop_button = ttk.Button(
            control_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED
        )
        self.stop_button.grid(row=13, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=14, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )
        
        # Stats display
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="10")
        stats_frame.grid(row=15, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=12, width=25, font=("Courier", 9))
        self.stats_text.pack()
        self.update_stats_display()
        
        # Right panel - Video display
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        video_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        self.video_label_widget = ttk.Label(video_frame)
        self.video_label_widget.pack()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def update_direction_options(self):
        """
        Update available direction based on selected orientation
        
        Horizontal lines: 'up' or 'down'
        Vertical lines: 'left' or 'right'
        """

        if self.orientation.get() == "horizontal":
            self.direction_menu['values'] = ["up", "down"]
            self.count_direction.set("down")
        else:
            self.direction_menu['values'] = ["left", "right"]
            self.count_direction.set("right")
    
    def update_position_label(self, value):
        """
        Update position label display when slider changes
        
        Args:
            value: New slider value.
        """

        self.position_label.config(text=f"{float(value):.2f}")
    
    def update_age_label(self, value):
        """
        Update track age label display when slider changes
        
        Args:
            value: New slider value
        """

        self.age_label.config(text=f"{int(float(value))}")
    
    def select_video(self):
        """
        Open file dialog to select a video file
        
        Updates video path and enables start button if successful
        """
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.video_path = Path(filename)
            self.video_label.config(text=self.video_path.name)
            self.start_button.config(state=tk.NORMAL)
            self.status_var.set(f"Video loaded: {self.video_path.name}")
    
    def start_processing(self):
        """
        Start video processing in a separate thread

        Initializes tracker and counter with set config
        Spawns background thread for video processing
        """
        if not self.video_path or self.is_playing:
            return
        
        self.is_playing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Processing...")
        
        self.tracker = PersonTracker(
            model_name="models/ft_yolo11s.pt",
            confidence_threshold=0.35,
            iou_threshold=0.05
        )
        
        self.counter = DirectionalCounter(
            line_position=self.line_position.get(),
            orientation=self.orientation.get(),
            count_direction=self.count_direction.get(),
            min_track_age=self.min_track_age.get()
        )
        
        thread = threading.Thread(target=self.process_video, daemon=True)
        thread.start()
    
    def stop_processing(self):
        """
        Stop video processing and release resources
        
        Sets flag to stop processing loop and releases video capture
        """

        self.is_playing = False
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        self.status_var.set("Stopped")
        
        if self.cap:
            self.cap.release()
    
    def process_video(self):
        """
        Process video frames in background thread
        
        For each frame:
        - Runs detection and tracking
        - Checks for crossings
        - Draws visualization overlay
        - Updates GUI display
        - Updates statistics
        
        Runs until video completes or user stops processing
        """

        try:
            self.cap = cv2.VideoCapture(str(self.video_path))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            frame_delay = int(1000 / fps) if fps > 0 else 33
            
            for detections in self.tracker.track_video(self.video_path, return_frames=True):
                if not self.is_playing:
                    break
                
                for person in detections.persons:
                    trajectory = self.tracker.get_track_trajectory(person.track_id)
                    normalized_trajectory = [(x / w, y / h) for x, y in trajectory]
                    self.counter.check_crossing(person.track_id, normalized_trajectory)
                
                if detections.frame_image is not None:
                    frame = detections.frame_image.copy()
                    
                    if self.orientation.get() == 'horizontal':
                        y = int(h * self.line_position.get())
                        cv2.line(frame, (0, y), (w, y), (0, 0, 255), 3)
                    else:
                        x = int(w * self.line_position.get())
                        cv2.line(frame, (x, 0), (x, h), (0, 0, 255), 3)
                    
                    frame = draw_detections(frame, detections)
                    
                    self.display_frame(frame)
                    
                    self.update_stats_display()
                
                self.root.after(frame_delay)
            
            self.is_playing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("Processing complete")
            
            self.print_ground_truth_comparison()
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.is_playing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        
        finally:
            if self.cap:
                self.cap.release()
    
    def display_frame(self, frame):
        """
        Display video frame in GUI widget

        Resizes frame to fit display area and converts BGR to RGB
        
        Args:
            frame: Video frame to display
        """

        display_width = 900
        h, w = frame.shape[:2]
        scale = display_width / w
        display_height = int(h * scale)
        
        frame_resized = cv2.resize(frame, (display_width, display_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label_widget.imgtk = imgtk
        self.video_label_widget.configure(image=imgtk)
    
    def print_ground_truth_comparison(self):
        """
        Print ground truth comparison to console

        Loads ground truth and compares predicted counts with ground truth
        Prints accuracy metrics and directional breakdowns
        """

        if not self.video_path:
            return
        
        video_name = self.video_path.stem
        gt_dir = self.video_path.parent.parent / "ground_truth"
        
        if not gt_dir.exists():
            return
        
        gt_file = gt_dir / f"{video_name}.xgtf"
        if not gt_file.exists():
            return
        
        try:
            from utils.xgtf_parser import parse_xgtf
            ground_truth = parse_xgtf(gt_file)
            
            stats = self.counter.get_stats()
            metrics = self.tracker.get_metrics(min_track_length=10)
            direction_totals = ground_truth.get_direction_totals()
            
            predicted_count = stats['total_count']
            gt_total_crosses = ground_truth.total_crossings
            error = stats['total_crosses'] - gt_total_crosses
            accuracy = 100 * (1 - abs(error) / max(gt_total_crosses, 1))
            
            print("\nGround Truth Comparison:")
            print(f"\nVideo: {self.video_path.name}")
            print(f"Ground Truth File: {gt_file.name}")
            print()
            print(f"Predicted Net Count: {predicted_count}")
            print(f"Predicted Crosses:   {stats['total_crosses']}")
            print(f"GT Total Crosses:    {gt_total_crosses}")
            print(f"Error:               {error:+d}")
            print(f"Accuracy:            {accuracy:.1f}%")
            print()
            print(f"GT By Direction:")
            for direction, count in direction_totals.items():
                print(f"  {direction}: {count}")
            print()
            print(f"Predicted Tracks:    {metrics.get('valid_tracks', 0)}\n")
            
        except Exception as e:
            print(f"Could not load ground truth: {e}")
    
    def update_stats_display(self):
        """
        Update statistics display in left control panel
        
        Shows:
        - Current count and crosses
        - Processing metrics (FPS, time per frame)
        - Track statistics (total, valid, short tracks)
        - Track length metrics
        """

        if self.counter and self.tracker:
            stats = self.counter.get_stats()
            metrics = self.tracker.get_metrics(min_track_length=10)
            
            # Must be formatted like this
            stats_str = f"""
COUNT: {stats['total_count']}
CROSSES: {stats['total_crosses']}
COUNTED: {stats['unique_counted_tracks']}

FPS: {metrics.get('fps', 0):.1f}
TIME/FRAME: {metrics.get('avg_time_per_frame', 0)*1000:.1f}ms

TOTAL TRACKS: {metrics.get('total_tracks', 0)}
VALID TRACKS: {metrics.get('valid_tracks', 0)}
SHORT TRACKS: {metrics.get('short_tracks', 0)}

AVG LENGTH: {metrics.get('avg_track_length', 0):.1f}
MAX LENGTH: {metrics.get('max_track_length', 0)}
            """.strip()
        else:
            stats_str = """
COUNT: 0
CROSSES: 0
COUNTED: 0

FPS: 0.0
TIME/FRAME: 0.0ms

TOTAL TRACKS: 0
VALID TRACKS: 0
SHORT TRACKS: 0

AVG LENGTH: 0.0
MAX LENGTH: 0
            """.strip()
        
        # Only update if stats have changed
        if stats_str != self.last_stats_str:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_str)
            self.last_stats_str = stats_str


def main():
    """
    Run the GUI application
    
    Creates Tkinter root window and starts event loop
    """
    root = tk.Tk()
    app = CounterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
