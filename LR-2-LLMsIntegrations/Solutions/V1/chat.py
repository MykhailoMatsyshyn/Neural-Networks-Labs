import tkinter as tk
from tkinter import ttk, messagebox
from openai import OpenAI
import openai
import threading
import json
import os
import time
import statistics
from collections import Counter

# Initialize the main Tkinter window
root = tk.Tk()
root.title("LM Studio Chat Interface")
root.geometry("650x650")
root.configure(bg="#e6f3fa")

# Apply a modern theme with custom styles
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Arial", 13, "bold"), padding=12, background="#0078d4", foreground="white")
style.map("TButton", background=[("active", "#005a9e")])
style.configure("TLabel", font=("Arial", 13), background="#e6f3fa", foreground="#333333")
style.configure("TEntry", font=("Arial", 13), fieldbackground="white")
style.configure("TNotebook", background="#ffffff", tabmargins=5)
style.configure("TNotebook.Tab", font=("Arial", 13, "bold"), padding=[10, 5], foreground="#005a9e")
style.configure("TFrame", background="#ffffff")

# Initialize OpenAI client with the local server
client = OpenAI(base_url="http://192.168.0.105:1234/v1", api_key="lm-studio")

# Function to handle the API request in a separate thread
def send_request():
    # Get input values from the GUI
    system_prompt = system_entry.get() or "You are a helpful assistant."
    prompt = prompt_text.get("1.0", tk.END).strip()
    try:
        temperature = float(temperature_entry.get())
        max_tokens = int(max_tokens_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Temperature and Max Tokens must be valid numbers!", parent=chat_frame)
        return

    # Validate inputs
    if not prompt:
        messagebox.showerror("Input Error", "Prompt cannot be empty!", parent=chat_frame)
        return

    # Show loader
    response_id_text.config(state="normal")
    response_id_text.delete("1.0", tk.END)
    response_id_text.insert(tk.END, "Processing...")
    response_id_text.config(state="disabled", foreground="#ff0000")
    result_text.config(state="normal")
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, "Processing...")
    result_text.config(state="disabled", foreground="#ff0000")
    submit_button.config(state="disabled")

    # Function to perform the API call
    def perform_request():
        try:
            # Create chat completion using OpenAI client
            completion = client.chat.completions.create(
                model="mistralai/mistral-7b-instruct-v0.3",
                messages=[
                    {"role": "user", "content": f"{system_prompt} {prompt}"}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract response text
            output_text = completion.choices[0].message.content if completion.choices else "No output"

            # Update GUI with the result
            root.after(0, lambda: update_result(output_text, completion.id, system_prompt, prompt, temperature, max_tokens))

            # Save dialog to a file
            with open("dialog_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"System: {system_prompt}\n")
                f.write(f"Temperature: {temperature}\n")
                f.write(f"Max Tokens: {max_tokens}\n")
                f.write(f"Output: {output_text}\n")
                f.write(f"---\n")

        except openai.APIError as e:
            # Handle API-related errors
            root.after(0, lambda: show_error(f"API Error: {e}", e.response.status_code if hasattr(e, "response") else None, e.response.text if hasattr(e, "response") else None))
        except openai.OpenAIError as e:
            # Handle other OpenAI-specific errors
            root.after(0, lambda: show_error(f"OpenAI Error: {e}"))
        except Exception as e:
            # Handle unexpected errors
            root.after(0, lambda: show_error(f"An unexpected error occurred: {e}"))

    # Run the API call in a separate thread to avoid freezing the GUI
    threading.Thread(target=perform_request, daemon=True).start()

# Function to update the result text in the GUI
def update_result(output_text, response_id, system_prompt, prompt, temperature, max_tokens):
    response_id_text.config(state="normal")
    response_id_text.delete("1.0", tk.END)
    response_id_text.insert(tk.END, f"{response_id}")
    response_id_text.config(state="disabled", foreground="#333333")
    result_text.config(state="normal")
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, output_text)
    result_text.config(state="disabled", foreground="#333333")
    submit_button.config(state="normal")
    update_history()  # Update history after new dialog is saved

# Function to show error messages
def show_error(error_message, status_code=None, response_text=None):
    messagebox.showerror("Error", f"{error_message}\nStatus Code: {status_code or 'N/A'}\nResponse: {response_text or 'N/A'}", parent=chat_frame)
    response_id_text.config(state="normal")
    response_id_text.delete("1.0", tk.END)
    response_id_text.insert(tk.END, "Error occurred")
    response_id_text.config(state="disabled", foreground="#333333")
    result_text.config(state="normal")
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, "Check the error message.")
    result_text.config(state="disabled", foreground="#333333")
    submit_button.config(state="normal")

# Function to read and display history from dialog_log.txt
def update_history():
    history_text.config(state="normal")
    history_text.delete("1.0", tk.END)
    try:
        if os.path.exists("dialog_log.txt"):
            with open("dialog_log.txt", "r", encoding="utf-8") as f:
                entries = f.read().strip().split("---\n")
                for entry in entries:
                    if entry.strip():
                        lines = entry.strip().split("\n")
                        entry_dict = {}
                        current_key = None
                        current_value = []
                        for line in lines:
                            if ": " in line and not current_key:
                                key, value = line.split(": ", 1)
                                entry_dict[key] = value
                                current_key = key
                                current_value = [value]
                            elif line.strip() == "":
                                if current_key:
                                    current_value.append("")
                            elif ": " in line:
                                if current_key:
                                    entry_dict[current_key] = "\n".join(current_value)
                                key, value = line.split(": ", 1)
                                entry_dict[key] = value
                                current_key = key
                                current_value = [value]
                            else:
                                if current_key:
                                    current_value.append(line)
                        if current_key:
                            entry_dict[current_key] = "\n".join(current_value)
                        # Format each entry with styled tags
                        history_text.insert(tk.END, "System: ", "header")
                        history_text.insert(tk.END, f"{entry_dict.get('System', 'N/A')}\n")
                        history_text.insert(tk.END, "Prompt: ", "header")
                        history_text.insert(tk.END, f"{entry_dict.get('Prompt', 'N/A')}\n")
                        history_text.insert(tk.END, "Temperature: ", "header")
                        history_text.insert(tk.END, f"{entry_dict.get('Temperature', 'N/A')}\n")
                        history_text.insert(tk.END, "Max Tokens: ", "header")
                        history_text.insert(tk.END, f"{entry_dict.get('Max Tokens', 'N/A')}\n")
                        history_text.insert(tk.END, "Output: ", "header")
                        history_text.insert(tk.END, f"{entry_dict.get('Output', 'N/A')}\n\n")
                        history_text.insert(tk.END, "-" * 60 + "\n", "separator")
    except Exception as e:
        history_text.insert(tk.END, f"Error reading history: {e}")
    history_text.config(state="disabled")

# Function to display analytics
def display_analytics():
    analytics_text.config(state="normal")
    analytics_text.delete("1.0", tk.END)
    try:
        if os.path.exists("dialog_log.txt"):
            with open("dialog_log.txt", "r", encoding="utf-8") as f:
                entries = f.read().strip().split("---\n")
                if not entries or not entries[0].strip():
                    analytics_text.insert(tk.END, "No history available.")
                    analytics_text.config(state="disabled")
                    return
                temperatures = []
                max_tokens = []
                system_prompts = []
                for entry in entries:
                    if entry.strip():
                        lines = entry.strip().split("\n")
                        entry_dict = {}
                        current_key = None
                        current_value = []
                        for line in lines:
                            if ": " in line and not current_key:
                                key, value = line.split(": ", 1)
                                entry_dict[key] = value
                                current_key = key
                                current_value = [value]
                            elif line.strip() == "":
                                if current_key:
                                    current_value.append("")
                            elif ": " in line:
                                if current_key:
                                    entry_dict[current_key] = "\n".join(current_value)
                                key, value = line.split(": ", 1)
                                entry_dict[key] = value
                                current_key = key
                                current_value = [value]
                            else:
                                if current_key:
                                    current_value.append(line)
                        if current_key:
                            entry_dict[current_key] = "\n".join(current_value)
                        if "Temperature" in entry_dict:
                            try:
                                temperatures.append(float(entry_dict["Temperature"]))
                            except ValueError:
                                pass
                        if "Max Tokens" in entry_dict:
                            try:
                                max_tokens.append(int(entry_dict["Max Tokens"]))
                            except ValueError:
                                pass
                        if "System" in entry_dict:
                            system_prompts.append(entry_dict["System"])
                num_requests = len(entries)
                avg_temperature = statistics.mean(temperatures) if temperatures else 0
                avg_max_tokens = statistics.mean(max_tokens) if max_tokens else 0
                system_prompt_counts = Counter(system_prompts)
                analytics_text.insert(tk.END, f"Total Requests: {num_requests}\n", "header")
                analytics_text.insert(tk.END, f"Average Temperature: {avg_temperature:.2f}\n", "header")
                analytics_text.insert(tk.END, f"Average Max Tokens: {avg_max_tokens:.0f}\n", "header")
                analytics_text.insert(tk.END, "System Prompt Distribution:\n", "header")
                for prompt, count in system_prompt_counts.items():
                    analytics_text.insert(tk.END, f"{prompt}: {count} times\n")
                analytics_text.insert(tk.END, "-" * 60 + "\n", "separator")
    except Exception as e:
        analytics_text.insert(tk.END, f"Error analyzing history: {e}")
    analytics_text.config(state="disabled")

# Function to periodically check for file updates
def check_file_update():
    global last_modified
    try:
        if os.path.exists("dialog_log.txt"):
            current_modified = os.path.getmtime("dialog_log.txt")
            if current_modified != last_modified:
                last_modified = current_modified
                update_history()
                display_analytics()
    except Exception:
        pass
    root.after(2000, check_file_update)  # Check every 2 seconds

# Initialize last modified time for dialog_log.txt
last_modified = 0 if not os.path.exists("dialog_log.txt") else os.path.getmtime("dialog_log.txt")

# Configure text tags for history and analytics styling
def configure_text_tags():
    history_text.tag_configure("header", font=("Arial", 13, "bold"), foreground="#005a9e")
    history_text.tag_configure("separator", foreground="#dcdcdc")
    analytics_text.tag_configure("header", font=("Arial", 13, "bold"), foreground="#005a9e")
    analytics_text.tag_configure("separator", foreground="#dcdcdc")

# Create notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=15, pady=15)

# Chat tab
chat_frame = ttk.Frame(notebook, padding=20, relief="ridge", borderwidth=2)
notebook.add(chat_frame, text="Chat")

# History tab
history_frame = ttk.Frame(notebook, padding=20, relief="ridge", borderwidth=2)
notebook.add(history_frame, text="History")

# Analytics tab
analytics_frame = ttk.Frame(notebook, padding=20, relief="ridge", borderwidth=2)
notebook.add(analytics_frame, text="Analytics")

# Create GUI elements for Chat tab
ttk.Label(chat_frame, text="System Prompt:", font=("Arial", 13, "bold")).pack(pady=8)
system_entry = ttk.Entry(chat_frame, width=50)
system_entry.insert(0, "You are a helpful assistant.")
system_entry.pack(pady=5)

ttk.Label(chat_frame, text="Temperature (0.0-2.0):", font=("Arial", 13, "bold")).pack(pady=8)
temperature_entry = ttk.Entry(chat_frame, width=10)
temperature_entry.insert(0, "0.7")
temperature_entry.pack(pady=5)

ttk.Label(chat_frame, text="Max Tokens:", font=("Arial", 13, "bold")).pack(pady=8)
max_tokens_entry = ttk.Entry(chat_frame, width=10)
max_tokens_entry.insert(0, "500")  # Increased default to 500
max_tokens_entry.pack(pady=5)

ttk.Label(chat_frame, text="Prompt:", font=("Arial", 13, "bold")).pack(pady=8)
prompt_text = tk.Text(chat_frame, height=5, width=50, font=("Arial", 13), borderwidth=2, relief="groove", bg="white")
prompt_text.pack(pady=5)

submit_button = ttk.Button(chat_frame, text="Submit", command=send_request)
submit_button.pack(pady=15)

ttk.Label(chat_frame, text="Response ID:", font=("Arial", 13, "bold")).pack(pady=8)
response_id_text = tk.Text(chat_frame, height=1, width=50, font=("Arial", 13, "bold"), borderwidth=2, relief="groove", bg="white")
response_id_text.pack(pady=5)

ttk.Label(chat_frame, text="Result:", font=("Arial", 13, "bold")).pack(pady=8)
result_scroll = ttk.Scrollbar(chat_frame, orient="vertical")
result_scroll.pack(side="right", fill="y")
result_text = tk.Text(chat_frame, height=8, width=50, font=("Arial", 13), borderwidth=2, relief="groove", bg="white", yscrollcommand=result_scroll.set)
result_text.pack(pady=5)
result_scroll.config(command=result_text.yview)

# Create GUI elements for History tab
ttk.Label(history_frame, text="Chat History:", font=("Arial", 14, "bold")).pack(pady=8)
history_scroll = ttk.Scrollbar(history_frame, orient="vertical")
history_scroll.pack(side="right", fill="y")
history_text = tk.Text(history_frame, height=20, width=60, font=("Arial", 13), borderwidth=2, relief="groove", bg="white", yscrollcommand=history_scroll.set)
history_text.pack(fill="both", expand=True, pady=5)
history_scroll.config(command=history_text.yview)

# Create GUI elements for Analytics tab
ttk.Label(analytics_frame, text="Analytics:", font=("Arial", 14, "bold")).pack(pady=8)
analytics_scroll = ttk.Scrollbar(analytics_frame, orient="vertical")
analytics_scroll.pack(side="right", fill="y")
analytics_text = tk.Text(analytics_frame, height=20, width=60, font=("Arial", 13), borderwidth=2, relief="groove", bg="white", yscrollcommand=analytics_scroll.set)
analytics_text.pack(fill="both", expand=True, pady=5)
analytics_scroll.config(command=analytics_text.yview)

# Configure text tags and initial content
configure_text_tags()
update_history()
display_analytics()

# Start periodic file checking
root.after(2000, check_file_update)

# Start the Tkinter event loop
root.mainloop()