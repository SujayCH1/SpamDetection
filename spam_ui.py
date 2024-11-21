import tkinter as tk
from tkinter import messagebox, ttk
from spam_detection_model import predict_spam

def main():
    def check_spam():
        email_text = email_input.get("1.0", tk.END).strip()
        if not email_text:
            messagebox.showwarning("Input Error", "Please enter an email to check.")
            return
        result = predict_spam(email_text)
        result_label.config(text=f"This email is: {result}", fg="green" if result == "Ham" else "red")
    
    root = tk.Tk()
    root.title("Spam Detection")
    root.geometry("500x400")
    root.resizable(False, False)
    
    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=5)
    style.configure("TLabel", font=("Helvetica", 12), padding=5)
    style.configure("TText", font=("Helvetica", 12))
    

    tk.Label(root, text="Spam Detection System", font=("Helvetica", 16, "bold"), pady=10).pack()
    
    tk.Label(root, text="Enter email text below:", font=("Helvetica", 12)).pack(pady=5)
    
    email_input = tk.Text(root, height=10, width=60, font=("Helvetica", 12))
    email_input.pack(pady=10)
    
    predict_button = ttk.Button(root, text="Check for Spam", command=check_spam)
    predict_button.pack(pady=10)
    
    result_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"))
    result_label.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()
