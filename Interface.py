import tkinter as tk
from tkinter import scrolledtext
import socket
import threading

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Общий чат")
        self.root.geometry("400x300")

        # Создание текстового поля для отображения сообщений
        self.text_area = scrolledtext.ScrolledText(self.root, width=50, height=10)
        self.text_area.pack(padx=10, pady=10)

        # Создание поля для ввода сообщений
        self.entry = tk.Entry(self.root, width=40)
        self.entry.pack(padx=10, pady=10)

        # Создание кнопки для отправки сообщений
        self.send_button = tk.Button(self.root, text="Отправить", command=self.send_message)
        self.send_button.pack(padx=10, pady=10)

        # Создание сокета для соединения с сервером
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(("localhost", 12345))

        # Создание потока для приема сообщений
        self.receive_thread = threading.Thread(target=self.receive_message)
        self.receive_thread.start()

    def send_message(self):
        message = self.entry.get()
        self.client_socket.send(message.encode("utf-8"))
        self.entry.delete(0, tk.END)

    def receive_message(self):
        while True:
            message = self.client_socket.recv(1024).decode("utf-8")
            self.text_area.insert(tk.END, message + "\n")
            self.text_area.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()

from flask import Flask
from flask import request
from threading import Thread
import time
import requests


app = Flask('')

@app.route('/')
def home():
  return "I'm alive"

def run():
  app.run(host='0.0.0.0', port=80)

def keep_alive():
  t = Thread(target=r