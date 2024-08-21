
import os
import hypertunnel

# Set the folder path and MAC addresses
folder_path = '/Users/sveta/Desktop/QueensCorp'
mac_address1 = '570e03ebe6d5'
mac_address2 = '254:142:10:11:94:26:22'

# Create a hypertunnel instance
tunnel = hypertunnel.Tunnel()

# Share the folder using the MAC addresses
tunnel.share_folder(folder_path, [mac_address1, mac_address2])

print(f'Folder shared successfully at {mac_address1} and {mac_address2}')
import socket
import threading

class ChatServer:
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("localhost", 12345))
        self.server_socket.listen(5)
        self.clients = []

    def handle_client(self, client_socket):
        while True:
            message = client_socket.recv(1024).decode("utf-8")
            for client in self.clients:
                client.send(message.encode("utf-8"))

    def start(self):
        while True:
            client_socket, address = self.server_socket.accept()
            self.clients.append(client_socket)
            client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
            client_thread.start()

if __name__ == "__main__":
    server = ChatServer()
    server.start()