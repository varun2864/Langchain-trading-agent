import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import random
import math
import requests
import json
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION & THEME
# ============================================================================

BACKEND_URL = "http://localhost:8000"

class Theme:
    # Dark, cyber-security inspired palette
    BG_DARKEST = "#020408"
    BG_PRIMARY = "#0a0e14"
    BG_SECONDARY = "#0d1117"
    BG_TERTIARY = "#161b22"
    BG_CARD = "#1c2128"
    BG_ELEVATED = "#262c36"
    
    CYAN = "#00d9ff"
    BLUE = "#58a6ff"
    GREEN = "#3fb950"
    RED = "#f85149"
    ORANGE = "#d29922"
    YELLOW = "#e3b341"
    PURPLE = "#a371f7"
    PINK = "#db61a2"
    
    TEXT_BRIGHT = "#f0f6fc"
    TEXT_PRIMARY = "#c9d1d9"
    TEXT_SECONDARY = "#8b949e"
    TEXT_MUTED = "#484f58"
    BORDER = "#30363d"
    
    FONT = "Segoe UI"
    MONO = "Consolas"

# ============================================================================
# BACKEND CLIENT
# ============================================================================

class BackendClient:
    @staticmethod
    def send_message(message, callback):
        def _run():
            try:
                response = requests.post(f"{BACKEND_URL}/chat", json={"message": message})
                if response.status_code == 200:
                    data = response.json()
                    callback(data.get("response"), None)
                else:
                    callback(None, f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                callback(None, f"Connection Error: {str(e)}")
        
        threading.Thread(target=_run, daemon=True).start()

    @staticmethod
    def upload_file(filepath, callback):
        def _run():
            try:
                with open(filepath, 'rb') as f:
                    files = {'file': f}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                
                if response.status_code == 200:
                    callback("File uploaded successfully.", None)
                else:
                    callback(None, f"Upload failed: {response.text}")
            except Exception as e:
                callback(None, f"Error: {str(e)}")
        
        threading.Thread(target=_run, daemon=True).start()

    @staticmethod
    def clear_memory(callback):
        def _run():
            try:
                response = requests.post(f"{BACKEND_URL}/clear")
                if response.status_code == 200:
                    callback("Memory cleared.", None)
                else:
                    callback(None, f"Clear failed: {response.text}")
            except Exception as e:
                callback(None, f"Error: {str(e)}")
        
        threading.Thread(target=_run, daemon=True).start()

# ============================================================================
# CHAT PANEL
# ============================================================================

class ChatPanel(tk.Frame):
    def __init__(self, parent, send_callback, **kwargs):
        super().__init__(parent, bg=Theme.BG_SECONDARY, **kwargs)
        self.send_callback = send_callback
        
        # Header
        header = tk.Frame(self, bg=Theme.BG_TERTIARY, height=40)
        header.pack(fill='x')
        header.pack_propagate(False)
        tk.Label(header, text="  COMMUNICATION LOG", font=(Theme.FONT, 10, 'bold'),
                fg=Theme.CYAN, bg=Theme.BG_TERTIARY).pack(side='left', pady=8)
        
        # Scrollable Container
        container = tk.Frame(self, bg=Theme.BG_SECONDARY)
        container.pack(fill='both', expand=True, padx=2, pady=2)
        
        self.canvas = tk.Canvas(container, bg=Theme.BG_PRIMARY, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient='vertical', command=self.canvas.yview,
                                bg=Theme.BG_TERTIARY, troughcolor=Theme.BG_PRIMARY, width=12)
        
        self.inner_frame = tk.Frame(self.canvas, bg=Theme.BG_PRIMARY)
        self.inner_frame.bind("<Configure>",
                             lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor='nw')
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Resize inner frame to match canvas width
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))
        
        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        
        # Input Area
        input_frame = tk.Frame(self, bg=Theme.BG_TERTIARY, pady=5, padx=5)
        input_frame.pack(fill='x')
        
        self.input_var = tk.StringVar()
        self.entry = tk.Entry(input_frame, textvariable=self.input_var, bg=Theme.BG_DARKEST,
                            fg=Theme.TEXT_BRIGHT, font=(Theme.FONT, 11), relief='flat', insertbackground='white')
        self.entry.pack(side='left', fill='x', expand=True, padx=(0, 5), ipady=5)
        self.entry.bind('<Return>', self._on_send)
        
        btn = tk.Button(input_frame, text="SEND", command=self._on_send,
                      bg=Theme.CYAN, fg=Theme.BG_DARKEST, font=(Theme.MONO, 9, 'bold'),
                      relief='flat', activebackground=Theme.BLUE)
        btn.pack(side='right')

    def _on_send(self, event=None):
        msg = self.input_var.get().strip()
        if msg:
            self.add_message("USER", msg, 'user')
            self.input_var.set("")
            self.send_callback(msg)

    def add_message(self, sender, text, tag):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Determine styles based on tag
        if tag == 'user':
            border_color = Theme.BLUE
            bg_color = Theme.BG_ELEVATED
            header_text = "USER"
            header_fg = Theme.BLUE
            align = 'e' # Right align for user? No, let's keep all left for consistency but style differently
        elif tag == 'agent':
            border_color = Theme.GREEN
            bg_color = Theme.BG_CARD
            header_text = "TRADING ASSISTANT"
            header_fg = Theme.GREEN
        elif tag == 'error':
            border_color = Theme.RED
            bg_color = Theme.BG_CARD
            header_text = "SYSTEM ERROR"
            header_fg = Theme.RED
        else: # system
            border_color = Theme.ORANGE
            bg_color = Theme.BG_CARD
            header_text = "SYSTEM ALERT"
            header_fg = Theme.ORANGE
            
        # Create Card
        outer = tk.Frame(self.inner_frame, bg=border_color, padx=1, pady=1)
        outer.pack(fill='x', padx=10, pady=6)
        
        inner = tk.Frame(outer, bg=bg_color, padx=10, pady=8)
        inner.pack(fill='x')
        
        # Header Row
        header_frame = tk.Frame(inner, bg=bg_color)
        header_frame.pack(fill='x', pady=(0, 4))
        
        tk.Label(header_frame, text=header_text, font=(Theme.MONO, 8, 'bold'),
                fg=header_fg, bg=bg_color).pack(side='left')
        
        tk.Label(header_frame, text=f"[{timestamp}]", font=(Theme.MONO, 8),
                fg=Theme.TEXT_MUTED, bg=bg_color).pack(side='right')
                
        # Message Body
        msg_label = tk.Label(inner, text=text, font=(Theme.FONT, 10),
                           fg=Theme.TEXT_BRIGHT, bg=bg_color, justify='left', anchor='w', wraplength=350)
        msg_label.pack(anchor='w', fill='x')
        
        # Auto-scroll
        self.inner_frame.update_idletasks()
        self.canvas.yview_moveto(1.0)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class TradingBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("AI TRADING INTERFACE")
        self.geometry("1000x800")
        self.configure(bg=Theme.BG_DARKEST)
        
        # Main Container
        main_container = tk.Frame(self, bg=Theme.BG_DARKEST)
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # ASCII Art Header
        ascii_art = r"""
  ______   ______        ________                        __  __                            _______               __     
 /      \ /      |      /        |                      /  |/  |                          /       \             /  |    
/$$$$$$  |$$$$$$/       $$$$$$$$/______   ______    ____$$ |$$/  _______    ______        $$$$$$$  |  ______   _$$ |_   
$$ |__$$ |  $$ |           $$ | /      \ /      \  /    $$ |/  |/       \  /      \       $$ |__$$ | /      \ / $$   |  
$$    $$ |  $$ |           $$ |/$$$$$$  |$$$$$$  |/$$$$$$$ |$$ |$$$$$$$  |/$$$$$$  |      $$    $$< /$$$$$$  |$$$$$$/   
$$$$$$$$ |  $$ |           $$ |$$ |  $$/ /    $$ |$$ |  $$ |$$ |$$ |  $$ |$$ |  $$ |      $$$$$$$  |$$ |  $$ |  $$ | __ 
$$ |  $$ | _$$ |_          $$ |$$ |     /$$$$$$$ |$$ \__$$ |$$ |$$ |  $$ |$$ \__$$ |      $$ |__$$ |$$ \__$$ |  $$ |/  |
$$ |  $$ |/ $$   |         $$ |$$ |     $$    $$ |$$    $$ |$$ |$$ |  $$ |$$    $$ |      $$    $$/ $$    $$/   $$  $$/ 
$$/   $$/ $$$$$$/          $$/ $$/       $$$$$$$/  $$$$$$$/ $$/ $$/   $$/  $$$$$$$ |      $$$$$$$/   $$$$$$/     $$$$/  
                                                                          /  \__$$ |                                    
                                                                          $$    $$/                                     
                                                                           $$$$$$/                                              
        """
        
        header_label = tk.Label(main_container, text=ascii_art, font=(Theme.MONO, 8), 
                              fg=Theme.CYAN, bg=Theme.BG_DARKEST, justify='left')
        header_label.pack(anchor='center', pady=(0, 20))
        
        # Controls
        controls = tk.Frame(main_container, bg=Theme.BG_ELEVATED, pady=10)
        controls.pack(fill='x', pady=(0, 10))
        
        tk.Button(controls, text="UPLOAD DOC", command=self._upload_doc,
                 bg=Theme.PURPLE, fg="white", font=(Theme.MONO, 9, 'bold'), relief='flat').pack(side='left', padx=10)
        
        tk.Button(controls, text="CLEAR MEMORY", command=self._clear_memory,
                 bg=Theme.RED, fg="white", font=(Theme.MONO, 9, 'bold'), relief='flat').pack(side='left')
        
        self.status_lbl = tk.Label(controls, text="SYSTEM READY", fg=Theme.GREEN, bg=Theme.BG_ELEVATED, font=(Theme.MONO, 9))
        self.status_lbl.pack(side='right', padx=10)
        
        # Chat
        self.chat = ChatPanel(main_container, self._handle_user_message)
        self.chat.pack(fill='both', expand=True)
        
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _handle_user_message(self, message):
        self.status_lbl.config(text="PROCESSING...", fg=Theme.YELLOW)
        BackendClient.send_message(message, self._on_agent_response)

    def _on_agent_response(self, response, error):
        self.status_lbl.config(text="SYSTEM READY", fg=Theme.GREEN)
        
        if error:
            self.chat.add_message("SYSTEM", error, 'error')
        else:
            self.chat.add_message("AGENT", response, 'agent')

    def _upload_doc(self):
        filepath = filedialog.askopenfilename(filetypes=[("PDF Documents", "*.pdf")])
        if filepath:
            self.chat.add_message("SYSTEM", f"Uploading: {os.path.basename(filepath)}...", 'system')
            BackendClient.upload_file(filepath, self._on_upload_complete)

    def _on_upload_complete(self, msg, error):
        if error:
            self.chat.add_message("SYSTEM", error, 'error')
        else:
            self.chat.add_message("SYSTEM", msg, 'system')

    def _clear_memory(self):
        BackendClient.clear_memory(self._on_clear_complete)

    def _on_clear_complete(self, msg, error):
        if error:
            self.chat.add_message("SYSTEM", error, 'error')
        else:
            self.chat.add_message("SYSTEM", msg, 'system')

    def _on_close(self):
        self.destroy()

if __name__ == "__main__":
    app = TradingBotGUI()
    app.mainloop()
