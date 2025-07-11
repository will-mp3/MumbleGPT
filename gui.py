import tkinter as tk
from tkinter import scrolledtext, messagebox, font
import torch
import threading
import sys
import os

# Import from your model file
from model import load_model, encode, decode, device


class RetroGPTGUI:
    def __init__(self):
        self.model = None
        self.is_generating = False
        self.max_tokens = 150
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("🖥️ GPT Terminal v1.0 - Neural Network Interface")
        self.root.geometry("900x700")
        self.root.configure(bg='#c0c0c0')  # Classic Mac gray
        
        # Set minimum window size
        self.root.minsize(600, 500)
        
        # Try to get retro fonts, fallback to system fonts
        try:
            self.font_family = 'Monaco' if 'Monaco' in font.families() else 'Courier'
        except:
            self.font_family = 'Courier'
        
        self.font_normal = (self.font_family, 11)
        self.font_title = (self.font_family, 14, 'bold')
        self.font_mono = (self.font_family, 10)
        
        self.setup_ui()
        self.setup_menu()
        self.load_model_async()
        
    def setup_menu(self):
        """Create a classic Mac-style menu bar"""
        menubar = tk.Menu(self.root, bg='#c0c0c0', fg='black')
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg='#c0c0c0', fg='black')
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Session", command=self.new_session)
        file_menu.add_command(label="Clear Output", command=self.clear_output)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0, bg='#c0c0c0', fg='black')
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Input", command=self.clear_input)
        edit_menu.add_command(label="Copy Output", command=self.copy_output)
        
        # Options menu
        options_menu = tk.Menu(menubar, tearoff=0, bg='#c0c0c0', fg='black')
        menubar.add_cascade(label="Options", menu=options_menu)
        options_menu.add_command(label="Settings", command=self.show_settings)
        
    def setup_ui(self):
        """Setup the main UI elements"""
        # Title frame with retro styling
        title_frame = tk.Frame(self.root, bg='#c0c0c0', relief='raised', bd=2)
        title_frame.pack(fill='x', padx=8, pady=8)
        
        title_label = tk.Label(title_frame, text="🖥️ GPT Neural Network Terminal", 
                              font=self.font_title, bg='#c0c0c0', fg='black')
        title_label.pack(side='left', pady=8)
        
        # Model status in title
        self.model_status = tk.Label(title_frame, text="Loading Model...", 
                                    font=self.font_normal, bg='#c0c0c0', fg='orange')
        self.model_status.pack(side='right', pady=8)
        
        # Main container with padding
        main_frame = tk.Frame(self.root, bg='#c0c0c0')
        main_frame.pack(fill='both', expand=True, padx=12, pady=8)
        
        # Input section
        input_frame = tk.Frame(main_frame, bg='#c0c0c0', relief='sunken', bd=2)
        input_frame.pack(fill='x', pady=(0, 12))
        
        # Input label and controls
        input_header = tk.Frame(input_frame, bg='#c0c0c0')
        input_header.pack(fill='x', padx=8, pady=(8, 0))
        
        input_label = tk.Label(input_header, text="Enter your prompt:", 
                              font=self.font_normal, bg='#c0c0c0', fg='black')
        input_label.pack(side='left')
        
        # Token count setting
        token_frame = tk.Frame(input_header, bg='#c0c0c0')
        token_frame.pack(side='right')
        
        tk.Label(token_frame, text="Max tokens:", font=self.font_normal, 
                bg='#c0c0c0', fg='black').pack(side='left')
        
        self.token_var = tk.StringVar(value="150")
        token_entry = tk.Entry(token_frame, textvariable=self.token_var, 
                              width=5, font=self.font_normal)
        token_entry.pack(side='left', padx=(5, 0))
        
        # Input text area
        input_container = tk.Frame(input_frame, bg='white', relief='sunken', bd=1)
        input_container.pack(fill='x', padx=8, pady=8)
        
        self.input_text = tk.Text(input_container, height=5, font=self.font_normal, 
                                 bg='white', fg='black', wrap=tk.WORD,
                                 relief='flat', bd=0, padx=4, pady=4)
        input_scrollbar = tk.Scrollbar(input_container, orient='vertical', 
                                      command=self.input_text.yview)
        self.input_text.configure(yscrollcommand=input_scrollbar.set)
        
        self.input_text.pack(side='left', fill='both', expand=True)
        input_scrollbar.pack(side='right', fill='y')
        
        # Button frame
        button_frame = tk.Frame(input_frame, bg='#c0c0c0')
        button_frame.pack(fill='x', padx=8, pady=8)
        
        # Generate button with retro styling
        self.generate_btn = tk.Button(button_frame, text="🚀 Generate", 
                                     command=self.generate_text,
                                     font=self.font_normal, bg='#d0d0d0', fg='black',
                                     relief='raised', bd=2, padx=20, pady=6,
                                     activebackground='#e0e0e0')
        self.generate_btn.pack(side='left')
        
        # Stop button
        self.stop_btn = tk.Button(button_frame, text="⏹️ Stop", 
                                 command=self.stop_generation,
                                 font=self.font_normal, bg='#ffcccc', fg='black',
                                 relief='raised', bd=2, padx=20, pady=6,
                                 activebackground='#ffdddd', state='disabled')
        self.stop_btn.pack(side='left', padx=(10, 0))
        
        # Status indicator
        self.status_frame = tk.Frame(button_frame, bg='#c0c0c0')
        self.status_frame.pack(side='right')
        
        self.status_label = tk.Label(self.status_frame, text="Ready", 
                                    font=self.font_normal, bg='#c0c0c0', fg='green')
        self.status_label.pack(side='right')
        
        # Progress indicator (simple dots)
        self.progress_label = tk.Label(self.status_frame, text="", 
                                      font=self.font_normal, bg='#c0c0c0', fg='orange')
        self.progress_label.pack(side='right', padx=(0, 10))
        
        # Output section
        output_frame = tk.Frame(main_frame, bg='#c0c0c0', relief='sunken', bd=2)
        output_frame.pack(fill='both', expand=True)
        
        output_header = tk.Frame(output_frame, bg='#c0c0c0')
        output_header.pack(fill='x', padx=8, pady=(8, 0))
        
        output_label = tk.Label(output_header, text="Terminal Output:", 
                               font=self.font_normal, bg='#c0c0c0', fg='black')
        output_label.pack(side='left')
        
        # Clear button for output
        clear_btn = tk.Button(output_header, text="Clear", 
                             command=self.clear_output,
                             font=self.font_normal, bg='#d0d0d0', fg='black',
                             relief='raised', bd=1, padx=10, pady=2)
        clear_btn.pack(side='right')
        
        # Output text area with retro terminal styling
        output_container = tk.Frame(output_frame, bg='#000000', relief='sunken', bd=1)
        output_container.pack(fill='both', expand=True, padx=8, pady=8)
        
        self.output_text = scrolledtext.ScrolledText(output_container, 
                                                    font=self.font_mono,
                                                    bg='#000000', fg='#00ff00',  # Green on black terminal
                                                    wrap=tk.WORD, relief='flat', bd=0,
                                                    padx=6, pady=6, state='disabled',
                                                    insertbackground='#00ff00')
        self.output_text.pack(fill='both', expand=True)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-Return>', lambda e: self.generate_text())
        self.root.bind('<Control-l>', lambda e: self.clear_output())
        self.root.bind('<Control-n>', lambda e: self.new_session())
        self.root.bind('<Escape>', lambda e: self.stop_generation())
        
        # Welcome message
        self.append_output("=" * 60)
        self.append_output("  GPT Neural Network Terminal v1.0")
        self.append_output("  Retro AI Interface - Classic Computing Style")
        self.append_output("=" * 60)
        self.append_output("")
        self.append_output("⚡ Loading neural network model...")
        self.append_output("Keyboard shortcuts:")
        self.append_output("   • Ctrl+Enter: Generate text")
        self.append_output("   • Ctrl+L: Clear output")
        self.append_output("   • Ctrl+N: New session")
        self.append_output("   • Escape: Stop generation")
        self.append_output("")
        self.append_output("Enter your prompt above and press Generate!")
        self.append_output("-" * 60)
        self.append_output("")
        
    def append_output(self, text):
        """Append text to the output area"""
        self.output_text.configure(state='normal')
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.configure(state='disabled')
        self.output_text.see(tk.END)
        
    def clear_output(self):
        """Clear the output area"""
        self.output_text.configure(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.configure(state='disabled')
        
    def clear_input(self):
        """Clear the input area"""
        self.input_text.delete(1.0, tk.END)
        
    def new_session(self):
        """Start a new session"""
        self.clear_output()
        self.clear_input()
        self.append_output("New session started")
        self.append_output("-" * 60)
        self.append_output("")
        
    def copy_output(self):
        """Copy output to clipboard"""
        self.root.clipboard_clear()
        self.root.clipboard_append(self.output_text.get(1.0, tk.END))
        self.status_label.configure(text="Copied to clipboard", fg='blue')
        self.root.after(2000, lambda: self.status_label.configure(text="Ready", fg='green'))
        
    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg='#c0c0c0')
        settings_window.resizable(False, False)
        
        # Center the window
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        tk.Label(settings_window, text="Terminal Settings", 
                font=self.font_title, bg='#c0c0c0').pack(pady=20)
        
        # Token settings
        token_frame = tk.Frame(settings_window, bg='#c0c0c0')
        token_frame.pack(pady=10)
        
        tk.Label(token_frame, text="Max tokens per generation:", 
                font=self.font_normal, bg='#c0c0c0').pack(side='left')
        
        token_scale = tk.Scale(token_frame, from_=50, to=500, orient='horizontal',
                              bg='#c0c0c0', font=self.font_normal)
        token_scale.set(self.max_tokens)
        token_scale.pack(side='left', padx=10)
        
        def apply_settings():
            self.max_tokens = token_scale.get()
            self.token_var.set(str(self.max_tokens))
            settings_window.destroy()
        
        tk.Button(settings_window, text="Apply", command=apply_settings,
                 font=self.font_normal, bg='#d0d0d0', relief='raised', bd=2).pack(pady=20)
        
    def load_model_async(self):
        """Load the model in a separate thread"""
        thread = threading.Thread(target=self._load_model_worker)
        thread.daemon = True
        thread.start()
        
    def _load_model_worker(self):
        """Worker function for loading the model"""
        try:
            self.model = load_model()
            self.root.after(0, self._model_loaded)
        except Exception as e:
            self.root.after(0, self._model_load_error, str(e))
            
    def _model_loaded(self):
        """Handle successful model loading"""
        self.model_status.configure(text="Model Ready", fg='green')
        self.append_output("Neural network model loaded successfully!")
        self.append_output(f"Device: {device}")
        self.append_output("Ready for text generation!")
        self.append_output("")
        
    def _model_load_error(self, error_msg):
        """Handle model loading errors"""
        self.model_status.configure(text="Model Error", fg='red')
        self.append_output(f"Error loading model: {error_msg}")
        self.append_output("Please check that 'model-01.pkl' and 'vocab.txt' exist")
        self.append_output("")
        
    def generate_text(self):
        """Generate text using the model"""
        if self.is_generating:
            return
            
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded! Please wait for model to load.")
            return
            
        prompt = self.input_text.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt first!")
            return
            
        try:
            self.max_tokens = int(self.token_var.get())
            if self.max_tokens < 1 or self.max_tokens > 1000:
                raise ValueError("Token count must be between 1 and 1000")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid token count: {e}")
            return
            
        # Start generation
        self.is_generating = True
        self.generate_btn.configure(state='disabled', text="Generating...")
        self.stop_btn.configure(state='normal')
        self.status_label.configure(text="Generating...", fg='orange')
        self.start_progress_animation()
        
        # Generate in separate thread
        thread = threading.Thread(target=self._generate_worker, args=(prompt,))
        thread.daemon = True
        thread.start()
        
    def start_progress_animation(self):
        """Start the progress dots animation"""
        self.progress_dots = 0
        self.animate_progress()
        
    def animate_progress(self):
        """Animate progress dots"""
        if self.is_generating:
            dots = "." * (self.progress_dots % 4)
            self.progress_label.configure(text=dots)
            self.progress_dots += 1
            self.root.after(500, self.animate_progress)
        else:
            self.progress_label.configure(text="")
            
    def stop_generation(self):
        """Stop the current generation"""
        self.is_generating = False
        self.generate_btn.configure(state='normal', text="🚀 Generate")
        self.stop_btn.configure(state='disabled')
        self.status_label.configure(text="Stopped", fg='red')
        
    def _generate_worker(self, prompt):
        """Worker function for text generation"""
        try:
            # Your original model code
            context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
            generated_chars = decode(self.model.generate(context.unsqueeze(0), max_new_tokens=self.max_tokens)[0].tolist())
            
            # Update GUI in main thread
            self.root.after(0, self._generation_complete, prompt, generated_chars)
            
        except Exception as e:
            self.root.after(0, self._generation_error, str(e))
            
    def _generation_complete(self, prompt, generated_text):
        """Handle completion of text generation"""
        if not self.is_generating:  # Check if stopped
            return
            
        self.append_output(f"PROMPT:")
        self.append_output(f"{prompt}")
        self.append_output("")
        self.append_output(f"COMPLETION:")
        self.append_output(f"{generated_text}")
        self.append_output("")
        self.append_output("=" * 60)
        self.append_output("")
        
        self.is_generating = False
        self.generate_btn.configure(state='normal', text="🚀 Generate")
        self.stop_btn.configure(state='disabled')
        self.status_label.configure(text="Complete", fg='green')
        
        # Auto-focus back to input
        self.input_text.focus_set()
        
    def _generation_error(self, error_msg):
        """Handle generation errors"""
        self.append_output(f"ERROR: {error_msg}")
        self.append_output("")
        self.append_output("-" * 60)
        self.append_output("")
        
        self.is_generating = False
        self.generate_btn.configure(state='normal', text="🚀 Generate")
        self.stop_btn.configure(state='disabled')
        self.status_label.configure(text="Error", fg='red')
        
    def run(self):
        """Start the GUI"""
        # Focus on input initially
        self.input_text.focus_set()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start the main loop
        self.root.mainloop()
        
    def on_closing(self):
        """Handle window closing"""
        if self.is_generating:
            if messagebox.askokcancel("Quit", "Generation in progress. Really quit?"):
                self.is_generating = False
                self.root.destroy()
        else:
            self.root.destroy()


# Main entry point
if __name__ == "__main__":
    try:
        # Create and run the GUI
        gui = RetroGPTGUI()
        gui.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting GUI: {e}")
        print("Make sure 'model-01.pkl' and 'vocab.txt' are in the same directory.")