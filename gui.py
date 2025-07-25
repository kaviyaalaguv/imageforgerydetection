import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
from fpdf import FPDF
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from datetime import datetime

class ForgeryDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🕵 Advanced Image Forgery Detection System")
        self.root.geometry("1200x900")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize variables
        self.original_path = ""
        self.altered_path = ""
        self.forgery_result_path = "forgery_result.jpg"
        self.results = None
        self.detection_complete = False
        
        # Create UI
        self.create_menu()
        self.create_main_ui()
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TButton", padding=6, relief="flat", background="#4CAF50", foreground="black")
        self.style.configure("TProgressbar", thickness=10, troughcolor="#f0f0f0", background="#4CAF50")
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to begin - please load images")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, 
                                 anchor=tk.W, background="#e0e0e0", foreground="#333333")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize matplotlib figure
        self.fig, self.ax = plt.subplots(1, 3, figsize=(12, 4))
        plt.close(self.fig)  # Close it to prevent empty window
    
    def validate_image_file(self, file_path):
        """Validates an image file can be opened and is supported."""
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify it's an image
            return True
        except Exception as e:
            messagebox.showerror("Invalid Image", 
                               f"Cannot open image file:\n{file_path}\n"
                               f"Error: {str(e)}")
            return False
    
    def convert_if_png(self, image_path):
        """Converts PNG to JPEG if necessary with better error handling."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        if image_path.lower().endswith(".png"):
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    jpeg_path = os.path.join(
                        os.path.dirname(image_path),
                        f"{os.path.splitext(os.path.basename(image_path))[0]}_{timestamp}.jpg"
                    )
                    img.save(jpeg_path, quality=95)
                    return jpeg_path
            except Exception as e:
                messagebox.showerror("Conversion Error", 
                                   f"Failed to convert PNG to JPG:\n{str(e)}\n"
                                   "Will attempt to use original PNG file.")
                return image_path
        return image_path
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Original Image", command=self.open_original_image)
        file_menu.add_command(label="Open Altered Image", command=self.open_altered_image)
        file_menu.add_separator()
        file_menu.add_command(label="Clear All", command=self.clear_all)
        file_menu.add_separator()
        file_menu.add_command(label="Save Report", command=self.save_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="Run Detection", command=self.run_detection)
        analysis_menu.add_command(label="Show Histogram Comparison", command=self.show_histograms)
        analysis_menu.add_command(label="Show Edge Comparison", command=self.show_edge_comparison)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_main_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with title
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, 
                              text="Digital Image Forgery Detection System", 
                              font=('Helvetica', 18, 'bold'))
        title_label.pack(side=tk.LEFT, padx=5)
        
        # Tool buttons frame
        tools_frame = ttk.Frame(main_frame)
        tools_frame.pack(fill=tk.X, pady=5)
        
        # Buttons with icons (using text as placeholders)
        self.open_orig_btn = ttk.Button(tools_frame, text="📂 Original Image", 
                                      command=self.open_original_image)
        self.open_orig_btn.pack(side=tk.LEFT, padx=5)
        
        self.open_alt_btn = ttk.Button(tools_frame, text="📂 Altered Image", 
                                     command=self.open_altered_image)
        self.open_alt_btn.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = ttk.Button(tools_frame, text="🔍 Analyze", 
                                     command=self.run_detection, 
                                     state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.report_btn = ttk.Button(tools_frame, text="📊 Save Report", 
                                    command=self.save_report, 
                                    state=tk.DISABLED)
        self.report_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(tools_frame, text="🔄 Clear All", 
                                  command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Content area with images and analysis
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Image display area (left)
        self.image_frame = ttk.Frame(content_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create a notebook for tabbed interface
        self.notebook = ttk.Notebook(self.image_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab for side-by-side comparison
        self.comparison_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_tab, text="Image Comparison")
        
        # Frame for image display
        self.img_display_frame = ttk.Frame(self.comparison_tab)
        self.img_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create placeholders for images
        self.original_label = ttk.Label(self.img_display_frame, 
                                      text="Original Image\nNot Loaded", 
                                      borderwidth=2, relief="groove")
        self.original_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.altered_label = ttk.Label(self.img_display_frame, 
                                     text="Altered Image\nNot Loaded", 
                                     borderwidth=2, relief="groove")
        self.altered_label.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.forgery_label = ttk.Label(self.img_display_frame, 
                                     text="Detection Result\nNot Available", 
                                     borderwidth=2, relief="groove")
        self.forgery_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # Configure grid weights
        self.img_display_frame.columnconfigure(0, weight=1)
        self.img_display_frame.columnconfigure(1, weight=1)
        self.img_display_frame.rowconfigure(0, weight=1)
        self.img_display_frame.rowconfigure(1, weight=1)
        
        # Tab for matplotlib visualization
        self.matplotlib_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.matplotlib_tab, text="Advanced Visualization")
        
        # Frame for matplotlib embedding
        self.plot_frame = ttk.Frame(self.matplotlib_tab)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Analysis panel (right)
        self.analysis_frame = ttk.LabelFrame(content_frame, text="Analysis Results", width=350)
        self.analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=5)
        
        # Results text area with scrollbar
        self.results_text = scrolledtext.ScrolledText(self.analysis_frame, width=45, height=30,
                                                     wrap=tk.WORD, font=('Consolas', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.insert(tk.END, "Upload images and run analysis to see results.")
        self.results_text.config(state=tk.DISABLED)
        
        # Progress bar at the bottom
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, 
                                      length=100, mode='determinate', 
                                      variable=self.progress_var)
        self.progress.pack(fill=tk.X, pady=(5, 0))
    
    def open_original_image(self):
        """Opens the original image with improved error handling."""
        file_path = filedialog.askopenfilename(
            title="Select Original Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            try:
                if not self.validate_image_file(file_path):
                    return
                
                self.status_var.set(f"Loading original image: {os.path.basename(file_path)}")
                self.root.update()
                
                self.original_path = self.convert_if_png(file_path)
                self.display_image(self.original_path, self.original_label, "Original Image")
                self.check_images_loaded()
                self.detection_complete = False
                self.report_btn.config(state=tk.DISABLED)
                self.status_var.set(f"Original image loaded: {os.path.basename(self.original_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}\n\nFile: {file_path}")
                self.status_var.set("Error loading original image")
    
    def open_altered_image(self):
        """Opens the altered image with improved error handling."""
        file_path = filedialog.askopenfilename(
            title="Select Altered Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            try:
                if not self.validate_image_file(file_path):
                    return
                
                self.status_var.set(f"Loading altered image: {os.path.basename(file_path)}")
                self.root.update()
                
                self.altered_path = self.convert_if_png(file_path)
                self.display_image(self.altered_path, self.altered_label, "Altered Image")
                self.check_images_loaded()
                self.detection_complete = False
                self.report_btn.config(state=tk.DISABLED)
                self.status_var.set(f"Altered image loaded: {os.path.basename(self.altered_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load altered image:\n{str(e)}\n\nFile: {file_path}\n"
                                           "Possible issues:\n"
                                           "- Corrupted image file\n"
                                           "- Unsupported format\n"
                                           "- Permission issues")
                self.status_var.set("Error loading altered image")
    
    def display_image(self, image_path, label, title):
        """Displays an image in the given label with better error handling."""
        try:
            # Open image and resize for display
            with Image.open(image_path) as img:
                img.thumbnail((400, 400))  # Resize to fit label
                
                # Create a frame for the image and title
                frame = ttk.Frame(label.master)
                frame.grid(row=label.grid_info()['row'], column=label.grid_info()['column'], 
                          sticky="nsew", padx=5, pady=5)
                
                # Display title
                title_label = ttk.Label(frame, text=title, font=('Helvetica', 10, 'bold'))
                title_label.pack(side=tk.TOP, fill=tk.X)
                
                # Display image
                photo = ImageTk.PhotoImage(img)
                img_label = ttk.Label(frame, image=photo)
                img_label.image = photo  # Keep a reference
                img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                
                # Replace the original label with our frame
                label.destroy()
                
        except Exception as e:
            label.config(image=None, text=f"Error loading image:\n{str(e)}")
            messagebox.showerror("Display Error", f"Cannot display image:\n{image_path}\nError: {str(e)}")
    
    def check_images_loaded(self):
        """Enables analysis button if both images are loaded."""
        if self.original_path and self.altered_path:
            self.analyze_btn.config(state=tk.NORMAL)
            self.status_var.set("Ready to analyze - click 'Analyze' button")
        else:
            self.analyze_btn.config(state=tk.DISABLED)
    
    def validate_images(self):
        """Validates that images are suitable for comparison."""
        if not self.original_path or not self.altered_path:
            messagebox.showwarning("Missing Images", "Please upload both images first.")
            return False
            
        try:
            # Load images with OpenCV
            original = cv2.imread(self.original_path)
            altered = cv2.imread(self.altered_path)
            
            if original is None or altered is None:
                messagebox.showerror("Error", "Failed to load one or both images")
                return False
                
            if original.shape != altered.shape:
                # Option to automatically resize
                if messagebox.askyesno("Size Mismatch", 
                                      "Images have different dimensions. Automatically resize the altered image?"):
                    altered = cv2.resize(altered, (original.shape[1], original.shape[0]))
                    cv2.imwrite(self.altered_path, altered)
                    self.display_image(self.altered_path, self.altered_label, "Altered Image (Resized)")
                    return True
                else:
                    return False
                    
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Image validation failed: {str(e)}")
            return False
    
    def run_detection(self):
        """Runs the forgery detection in a separate thread."""
        if not self.validate_images():
            return
            
        # Disable buttons during processing
        self.analyze_btn.config(state=tk.DISABLED)
        self.open_orig_btn.config(state=tk.DISABLED)
        self.open_alt_btn.config(state=tk.DISABLED)
        self.report_btn.config(state=tk.DISABLED)
        
        # Clear previous results
        self.detection_complete = False
        self.forgery_result_path = f"forgery_result_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        
        # Start progress animation
        self.progress_var.set(0)
        self.progress.start(10)
        self.status_var.set("Running forgery detection analysis...")
        
        # Update results text
        self.update_results_text("Running analysis...\n\nPlease wait...")
        
        # Run detection in a separate thread
        detection_thread = threading.Thread(target=self._run_detection_thread)
        detection_thread.daemon = True
        detection_thread.start()
    
    def _run_detection_thread(self):
        """Background thread for forgery detection."""
        try:
            # Load images
            original = cv2.imread(self.original_path)
            altered = cv2.imread(self.altered_path)
            
            # Ensure same dimensions
            if original.shape != altered.shape:
                altered = cv2.resize(altered, (original.shape[1], original.shape[0]))
                
            # Check if images are identical
            if np.array_equal(original, altered):
                self.root.after(0, lambda: messagebox.showinfo("Info", "Images are identical - no differences to detect"))
                self.root.after(0, self._reset_ui_after_detection)
                return
                
            # Convert to grayscale
            gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            gray_altered = cv2.cvtColor(altered, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray_original, gray_altered)
            
            # Apply threshold to highlight differences
            _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
            
            # Create color result (red highlights on original)
            result = original.copy()
            mask = threshold > 0
            result[mask] = [0, 0, 255]  # Set differences to red
            
            # Save the result
            cv2.imwrite(self.forgery_result_path, result)
            
            # Store results
            self.results = [original, altered, result]
            self.detection_complete = True
            
            # Update UI from main thread
            self.root.after(0, self._update_ui_after_detection)
            
        except Exception as e:
            self.root.after(0, lambda: self._show_error(f"Detection failed: {str(e)}"))
        finally:
            # Stop progress bar and re-enable buttons
            self.root.after(0, self._reset_ui_after_detection)
    
    def _update_ui_after_detection(self):
        """Update UI with detection results."""
        if self.results:
            # Display result image
            self.display_image(self.forgery_result_path, self.forgery_label, "Forgery Detection Result")
            
            # Create and display the matplotlib visualization
            self.show_comparison_plot()
            
            # Update analysis text
            self.update_analysis_text()
            
            # Enable report button
            self.report_btn.config(state=tk.NORMAL)
            
            # Set status
            self.status_var.set("Forgery detection completed successfully")
    
    def _reset_ui_after_detection(self):
        """Reset UI elements after detection."""
        self.progress.stop()
        self.progress_var.set(100)
        self.analyze_btn.config(state=tk.NORMAL)
        self.open_orig_btn.config(state=tk.NORMAL)
        self.open_alt_btn.config(state=tk.NORMAL)
    
    def _show_error(self, message):
        """Shows error message and resets UI."""
        messagebox.showerror("Error", message)
        self.status_var.set("Error: " + message)
        self.update_results_text(f"ERROR:\n{message}\n\nPlease try again with different images.")
    
    def show_comparison_plot(self):
        """Creates and displays a matplotlib comparison plot."""
        # Clear previous plot if exists
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        # Create a new figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle("Image Forgery Detection Analysis", fontsize=14)
        
        titles = ["Original Image", "Altered Image", "Forgery Detection"]
        images = [
            cv2.cvtColor(self.results[0], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(self.results[1], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(self.results[2], cv2.COLOR_BGR2RGB)
        ]
        
        for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
            ax.imshow(img)
            ax.set_title(title, fontsize=12)
            ax.axis("off")
            
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Switch to the matplotlib tab
        self.notebook.select(self.matplotlib_tab)
    
    def update_results_text(self, text):
        """Updates the results text area."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)
    
    def update_analysis_text(self):
        """Updates the analysis results text area with detailed report."""
        if not self.detection_complete:
            return
            
        # Calculate statistics
        forgery_percentage = self.calculate_forgery_percentage()
        forged_pixels = self.count_forged_pixels()
        confidence = self.calculate_confidence_level(forgery_percentage)
        
        # Get image properties
        orig_img = Image.open(self.original_path)
        width, height = orig_img.size
        file_size = self.get_file_size(self.original_path)
        img_format = os.path.splitext(self.original_path)[1].upper()[1:]
        
        # Current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build report
        report = f"""FORGERY DETECTION ANALYSIS REPORT
=================================
Date & Time: {current_time}

IMAGE INFORMATION
---------------------------------
Original Image: {os.path.basename(self.original_path)}
Altered Image: {os.path.basename(self.altered_path)}
Resolution: {width} x {height} pixels
Format: {img_format}
File Size: {file_size}

DETECTION RESULTS
---------------------------------
Forgery Status: {"FORGERY DETECTED" if forgery_percentage > 0.1 else "NO SIGNIFICANT ALTERATIONS"}
Altered Regions: {forgery_percentage:.2f}% of image
Total Forged Pixels: {forged_pixels:,}
Confidence Level: {confidence}

ANALYSIS DETAILS
---------------------------------
{self.get_analysis_details(forgery_percentage)}

RECOMMENDATION
---------------------------------
{self.get_recommendation(forgery_percentage)}

Note: This analysis is based on computer vision algorithms
and should be verified by a human expert for critical applications.
"""
        
        # Update the text widget
        self.update_results_text(report)
    
    def calculate_forgery_percentage(self):
        """Returns percentage of altered regions."""
        if not self.detection_complete or not os.path.exists(self.forgery_result_path):
            return 0.0
            
        # Calculate difference from result image
        result_img = cv2.imread(self.forgery_result_path)
        diff = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        
        total_pixels = threshold.size
        altered_pixels = np.count_nonzero(threshold)
        return (altered_pixels / total_pixels) * 100
    
    def count_forged_pixels(self):
        """Returns the number of forged pixels."""
        if not self.detection_complete or not os.path.exists(self.forgery_result_path):
            return 0
            
        result_img = cv2.imread(self.forgery_result_path)
        diff = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        return np.count_nonzero(threshold)
    
    def calculate_confidence_level(self, percentage):
        """Returns a confidence level based on percentage."""
        if percentage < 0.1:
            return "Very Low (likely no tampering)"
        elif percentage < 1:
            return "Low (minor alterations possible)"
        elif percentage < 5:
            return "Medium (some tampering detected)"
        elif percentage < 10:
            return "High (significant tampering)"
        else:
            return "Very High (extensive tampering)"
    
    def get_analysis_details(self, percentage):
        """Returns detailed analysis based on forgery percentage."""
        if percentage < 0.1:
            return ("The images appear nearly identical with minimal differences detected. "
                   "Any differences found are likely due to compression artifacts or minor "
                   "color variations rather than intentional tampering.")
        elif percentage < 1:
            return ("Minor alterations detected. These could represent small edits like "
                   "text additions, object removal, or localized adjustments. The changes "
                   "are not extensive but indicate some modification has occurred.")
        elif percentage < 5:
            return ("Moderate alterations detected. The image contains several modified "
                   "regions suggesting intentional editing. This could include object "
                   "addition/removal, background changes, or composite editing.")
        else:
            return ("Extensive alterations detected. The image has been significantly "
                   "modified with large areas changed. This suggests major editing like "
                   "composite images, extensive retouching, or complete background "
                   "replacement.")
    
    def get_recommendation(self, percentage):
        """Returns a recommendation based on forgery percentage."""
        if percentage < 0.1:
            return ("The image appears authentic with no significant signs of tampering. "
                   "No further action required unless other evidence suggests manipulation.")
        elif percentage < 1:
            return ("Minor alterations detected. Examine the highlighted areas closely "
                   "for signs of editing. Consider additional verification methods if "
                   "this image is critical to your work.")
        elif percentage < 5:
            return ("Moderate alterations detected. This image has likely been edited. "
                   "Treat its contents with caution and seek original/uncompressed versions "
                   "for verification if possible.")
        else:
            return ("Extensive alterations detected. This image has been significantly "
                   "modified and should not be considered authentic without additional "
                   "verification. Document all changes found for evidence.")
    
    def get_file_size(self, file_path):
        """Returns file size in human-readable format."""
        try:
            size_bytes = os.path.getsize(file_path)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.2f} TB"
        except:
            return "Unknown"
    
    def show_histograms(self):
        """Shows histogram comparison of original and altered images."""
        if not self.original_path or not self.altered_path:
            messagebox.showwarning("Missing Images", "Please upload both images first.")
            return
            
        try:
            # Load images
            original = cv2.imread(self.original_path)
            altered = cv2.imread(self.altered_path)
            
            # Create a new window
            hist_window = tk.Toplevel(self.root)
            hist_window.title("Image Histogram Comparison")
            hist_window.geometry("900x700")
            
            # Create figure with 3 rows (original, altered, difference)
            fig, axes = plt.subplots(3, 3, figsize=(12, 8))
            fig.suptitle("RGB Channel Histogram Comparison", fontsize=16)
            
            # Plot histograms for each channel
            colors = ('b', 'g', 'r')
            titles = ("Blue Channel", "Green Channel", "Red Channel")
            
            for i, color in enumerate(colors):
                # Original image histogram
                hist = cv2.calcHist([original], [i], None, [256], [0, 256])
                axes[0, i].plot(hist, color=color)
                axes[0, i].set_title(f"Original: {titles[i]}")
                axes[0, i].set_xlim([0, 256])
                
                # Altered image histogram
                hist = cv2.calcHist([altered], [i], None, [256], [0, 256])
                axes[1, i].plot(hist, color=color)
                axes[1, i].set_title(f"Altered: {titles[i]}")
                axes[1, i].set_xlim([0, 256])
                
                # Difference histogram
                diff = cv2.absdiff(original, altered)
                hist = cv2.calcHist([diff], [i], None, [256], [0, 256])
                axes[2, i].plot(hist, color=color)
                axes[2, i].set_title(f"Difference: {titles[i]}")
                axes[2, i].set_xlim([0, 256])
                
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=hist_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add close button
            close_btn = ttk.Button(hist_window, text="Close", command=hist_window.destroy)
            close_btn.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate histograms: {str(e)}")
    
    def show_edge_comparison(self):
        """Shows edge detection comparison of original and altered images."""
        if not self.original_path or not self.altered_path:
            messagebox.showwarning("Missing Images", "Please upload both images first.")
            return
            
        try:
            # Load images
            original = cv2.imread(self.original_path, cv2.IMREAD_GRAYSCALE)
            altered = cv2.imread(self.altered_path, cv2.IMREAD_GRAYSCALE)
            
            if original.shape != altered.shape:
                altered = cv2.resize(altered, (original.shape[1], original.shape[0]))
            
            # Apply Canny edge detection
            original_edges = cv2.Canny(original, 100, 200)
            altered_edges = cv2.Canny(altered, 100, 200)
            diff_edges = cv2.absdiff(original_edges, altered_edges)
            
            # Create a new window
            edge_window = tk.Toplevel(self.root)
            edge_window.title("Edge Detection Comparison")
            edge_window.geometry("900x700")
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle("Edge Detection Comparison", fontsize=16)
            
            # Plot images
            titles = ["Original Edges", "Altered Edges", "Edge Differences"]
            images = [original_edges, altered_edges, diff_edges]
            
            for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
                ax.imshow(img, cmap='gray')
                ax.set_title(title)
                ax.axis("off")
                
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=edge_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add close button
            close_btn = ttk.Button(edge_window, text="Close", command=edge_window.destroy)
            close_btn.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate edge comparison: {str(e)}")
    
    def save_report(self):
        """Generates and saves a detailed PDF report."""
        if not self.detection_complete:
            messagebox.showwarning("No Results", "Please run forgery detection first.")
            return
            
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile=f"forgery_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        )
        
        if not file_path:
            return  # User cancelled
            
        try:
            self.status_var.set("Generating PDF report...")
            self.root.update()
            
            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Add header
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Digital Image Forgery Detection Report", 0, 1, 'C')
            pdf.ln(5)
            
            # Add timestamp
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
            pdf.ln(10)
            
            # Section 1: Image Information
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "1. Image Information", 0, 1)
            pdf.set_font("Arial", '', 10)
            
            # Create a table for image info
            col_widths = [60, 120]
            pdf.cell(col_widths[0], 10, "Original Image:", 0, 0)
            pdf.cell(col_widths[1], 10, os.path.basename(self.original_path), 0, 1)
            
            pdf.cell(col_widths[0], 10, "Altered Image:", 0, 0)
            pdf.cell(col_widths[1], 10, os.path.basename(self.altered_path), 0, 1)
            
            pdf.cell(col_widths[0], 10, "Resolution:", 0, 0)
            img = Image.open(self.original_path)
            pdf.cell(col_widths[1], 10, f"{img.width} x {img.height} pixels", 0, 1)
            
            pdf.cell(col_widths[0], 10, "File Size:", 0, 0)
            pdf.cell(col_widths[1], 10, self.get_file_size(self.original_path), 0, 1)
            
            pdf.ln(10)
            
            # Section 2: Detection Results
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "2. Detection Results", 0, 1)
            pdf.set_font("Arial", '', 10)
            
            forgery_percentage = self.calculate_forgery_percentage()
            forged_pixels = self.count_forged_pixels()
            confidence = self.calculate_confidence_level(forgery_percentage)
            
            # Results table
            pdf.cell(col_widths[0], 10, "Forgery Status:", 0, 0)
            status = "FORGERY DETECTED" if forgery_percentage > 0.1 else "NO SIGNIFICANT ALTERATIONS"
            pdf.set_text_color(255, 0, 0) if status == "FORGERY DETECTED" else pdf.set_text_color(0, 128, 0)
            pdf.cell(col_widths[1], 10, status, 0, 1)
            pdf.set_text_color(0, 0, 0)
            
            pdf.cell(col_widths[0], 10, "Altered Regions:", 0, 0)
            pdf.cell(col_widths[1], 10, f"{forgery_percentage:.2f}% of image", 0, 1)
            
            pdf.cell(col_widths[0], 10, "Forged Pixels:", 0, 0)
            pdf.cell(col_widths[1], 10, f"{forged_pixels:,}", 0, 1)
            
            pdf.cell(col_widths[0], 10, "Confidence Level:", 0, 0)
            pdf.cell(col_widths[1], 10, confidence, 0, 1)
            
            pdf.ln(10)
            
            # Section 3: Visual Comparison
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "3. Visual Comparison", 0, 1)
            pdf.set_font("Arial", '', 10)
            
            # Add images to PDF
            try:
                # Create temporary resized versions for PDF
                img_orig = Image.open(self.original_path)
                img_orig.thumbnail((200, 200))
                orig_temp = "temp_orig.jpg"
                img_orig.save(orig_temp)
                
                img_alt = Image.open(self.altered_path)
                img_alt.thumbnail((200, 200))
                alt_temp = "temp_alt.jpg"
                img_alt.save(alt_temp)
                
                img_res = Image.open(self.forgery_result_path)
                img_res.thumbnail((200, 200))
                res_temp = "temp_res.jpg"
                img_res.save(res_temp)
                
                # Add images side by side
                pdf.cell(0, 10, "Original Image", 0, 1, 'C')
                pdf.image(orig_temp, x=30, w=50)
                pdf.cell(0, 10, "Altered Image", 0, 1, 'C')
                pdf.image(alt_temp, x=80, w=50)
                pdf.cell(0, 10, "Detection Result", 0, 1, 'C')
                pdf.image(res_temp, x=130, w=50)
                
                # Clean up temp files
                for temp_file in [orig_temp, alt_temp, res_temp]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
            except Exception as e:
                pdf.cell(0, 10, f"Error including images: {str(e)}", 0, 1)
            
            pdf.ln(10)
            
            # Section 4: Analysis
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "4. Analysis", 0, 1)
            pdf.set_font("Arial", '', 10)
            pdf.multi_cell(0, 10, self.get_analysis_details(forgery_percentage))
            pdf.ln(5)
            
            # Section 5: Recommendation
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "5. Recommendation", 0, 1)
            pdf.set_font("Arial", '', 10)
            pdf.multi_cell(0, 10, self.get_recommendation(forgery_percentage))
            pdf.ln(10)
            
            # Footer
            pdf.set_font("Arial", 'I', 8)
            pdf.cell(0, 10, "This report was generated by the Advanced Image Forgery Detection System", 0, 1, 'C')
            
            # Save PDF
            pdf.output(file_path)
            self.status_var.set(f"Report saved: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Report successfully saved to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
            self.status_var.set("Error generating report")
    
    def clear_all(self):
        """Clears all loaded images and results."""
        self.original_path = ""
        self.altered_path = ""
        self.forgery_result_path = "forgery_result.jpg"
        self.results = None
        self.detection_complete = False
        
        # Reset UI elements
        self.original_label.destroy()
        self.altered_label.destroy()
        self.forgery_label.destroy()
        
        # Recreate image labels
        self.original_label = ttk.Label(self.img_display_frame, 
                                      text="Original Image\nNot Loaded", 
                                      borderwidth=2, relief="groove")
        self.original_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.altered_label = ttk.Label(self.img_display_frame, 
                                     text="Altered Image\nNot Loaded", 
                                     borderwidth=2, relief="groove")
        self.altered_label.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.forgery_label = ttk.Label(self.img_display_frame, 
                                     text="Detection Result\nNot Available", 
                                     borderwidth=2, relief="groove")
        self.forgery_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # Clear results text
        self.update_results_text("Upload images and run analysis to see results.")
        
        # Reset buttons
        self.analyze_btn.config(state=tk.DISABLED)
        self.report_btn.config(state=tk.DISABLED)
        
        # Clear matplotlib tab
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        # Reset status
        self.status_var.set("Ready to begin - please load images")
    
    def show_user_guide(self):
        """Displays the user guide in a new window."""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide")
        guide_window.geometry("800x600")
        
        guide_text = """ADVANCED IMAGE FORGERY DETECTION SYSTEM - USER GUIDE

1. GETTING STARTED
------------------
- Use 'File > Open Original Image' to load the original reference image
- Use 'File > Open Altered Image' to load the potentially modified image
- Click the 'Analyze' button to run the forgery detection algorithm

2. UNDERSTANDING RESULTS
------------------------
- The system displays three views:
  * Original Image: The reference image you provided
  * Altered Image: The potentially modified image
  * Detection Result: Highlighted areas show detected differences
  
- The Analysis Results panel provides:
  * Percentage of altered regions
  * Total number of forged pixels
  * Confidence level assessment
  * Detailed analysis and recommendations

3. SAVING RESULTS
-----------------
- Use 'File > Save Report' to generate a comprehensive PDF report
- The report includes:
  * Image information and metadata
  * Visual comparison of images
  * Detailed analysis of detected alterations
  * Professional recommendations

4. ADVANCED FEATURES
--------------------
- View histogram comparisons under 'Analysis' menu
- Examine edge detection comparisons
- The system automatically handles:
  * Different image formats (JPEG, PNG, BMP)
  * Size mismatches (with user confirmation)
  * Color space conversions

BEST PRACTICES
--------------
- Use high-quality images for best results
- Ensure images are properly aligned before comparison
- Compare images with similar lighting conditions
- For critical applications, verify results with multiple methods

SUPPORT
-------
For assistance or to report issues, please contact:
support@forgerydetection.example.com
"""
        
        text_widget = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD, font=('Arial', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, guide_text)
        text_widget.config(state=tk.DISABLED)
        
        # Add close button
        close_btn = ttk.Button(guide_window, text="Close", command=guide_window.destroy)
        close_btn.pack(pady=10)
    
    def show_about(self):
        """Displays the about dialog."""
        about_text = """Advanced Image Forgery Detection System

Version: 2.1
Release Date: October 2023

Developed by: [Your Name]
Organization: [Your Organization]
License: MIT Open Source

This application uses advanced computer vision techniques to detect tampering in digital images. It compares an original image with a potentially altered version and highlights any differences with statistical analysis.

Features:
- Pixel-level difference detection
- Intelligent noise reduction
- Comprehensive reporting
- Multiple visualization methods

The system is designed for:
- Digital forensics
- Journalism verification
- Media authentication
- Academic research

For more information, please visit:
https://github.com/yourusername/forgery-detection
"""
        about_window = tk.Toplevel(self.root)
        about_window.title("About")
        about_window.geometry("500x400")
        
        text_widget = scrolledtext.ScrolledText(about_window, wrap=tk.WORD, font=('Arial', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, about_text)
        text_widget.config(state=tk.DISABLED)
        
        # Add close button
        close_btn = ttk.Button(about_window, text="Close", command=about_window.destroy)
        close_btn.pack(pady=10)
def show_user_feedback(altered_percentage):
    """Display feedback based on forgery results."""
    if altered_percentage > 50:
        messagebox.showinfo("Analysis Summary", "High probability of forgery detected! Consider verifying image authenticity.")
    elif 20 <= altered_percentage <= 50:
        messagebox.showinfo("Analysis Summary", "Moderate changes detected. Further validation recommended.")
    else:
        messagebox.showinfo("Analysis Summary", "Minimal alterations detected. Image appears authentic.")
def log_user_action(action):
    """Log user actions and results."""
    with open("user_activity.log", "a") as log_file:
        log_file.write(f"{action}\n")
# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = ForgeryDetectionApp(root)
    root.mainloop()