import socket 
import threading
import time 
from torchvision import models
import torch.nn as nn
import pickle
import codecs
import torch
import copy 
import numpy as np 
from io import BytesIO
import importlib
import torch.optim as optim 
from math import ceil
import argparse






class ConnectedClient():

    def __init__(self, id, conn):
        self.id = id
        self.conn = conn
        self.center_model = None
        self.center_optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialize()

    def initialize(self):

        self.center_model = MODEL.center(pretrained = True)
        self.center_optimizer = optim.Adam(self.center_model.parameters(), lr=LR)
        #send client_id 
        self.conn.sendall(f"{self.id}".encode())

    def forward_center(self):
        self.center_activations = self.center_model(self.front_activations)
        self.remote_center_activations = self.center_activations.detach().requires_grad_(True)


    def backward_center(self):
        self.center_activations.backward(self.remote_center_activations.grad)

    def get_front_activations(self):
        self.front_activations = self.receive_tensor()
        self.front_activations.requires_grad = True


    def send_center_activations(self):
        self.send_tensor(self.remote_center_activations)


    def get_center_activations_grad(self):
        self.remote_center_activations.grad = self.receive_tensor()


    def send_front_activations_grads(self):
        self.send_tensor(self.front_activations.grad)


    def send_tensor(self, arr):

        arr = arr.detach().numpy()
        f = BytesIO()
        np.savez(f, frame=arr)
        
        packet_size = len(f.getvalue())
        header = '{0}:'.format(packet_size)
        header = bytes(header.encode())  # prepend length of array

        out = bytearray()
        out += header

        f.seek(0)
        out += f.read()

        self.conn.sendall(out)


    def receive_tensor(self):

        length = None
        frameBuffer = bytearray()

        while True:
            data = self.conn.recv(1024)
            if len(data) == 0:
                return np.array([])
            frameBuffer += data
            if len(frameBuffer) == length:
                break
            while True:
                if length is None:
                    if b':' not in frameBuffer:
                        break
                    # remove the length bytes from the front of frameBuffer
                    # leave any remaining bytes in the frameBuffer!
                    length_str, ignored, frameBuffer = frameBuffer.partition(b':')
                    length = int(length_str)
                if len(frameBuffer) < length:
                    break
                # split off the full message from the remaining bytes
                # leave any remaining bytes in the frameBuffer!
                frameBuffer = frameBuffer[length:]
                length = None
                break
        
        arr = np.load(BytesIO(frameBuffer), allow_pickle=True)['frame']
        # print(f"CLient {self.id} frame received")
        return torch.from_numpy(arr)


def merge_weights(w):
    #after step op, merge weights 

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    
    return w_avg



def handle_client(c, client_id, lock):

    global models
    global clients_done

    conn_client = ConnectedClient(client_id, c)
    models[client_id] = conn_client.center_model
    num_iterations = ceil(100/BATCH_SIZE)
    num_test_iterations = ceil(50/BATCH_SIZE)


    for i in range(GLOBAL_EPOCHS):    #global epochs

        for iteration in range(num_iterations):

            conn_client.get_front_activations()
            conn_client.forward_center()
            conn_client.send_center_activations()
            conn_client.get_center_activations_grad()
            conn_client.backward_center()
            # conn_client.send_front_activations_grads()
            conn_client.center_optimizer.step()
            conn_client.center_optimizer.zero_grad()

        print(f'\rClient: {client_id} Local Epoch done: {i}\n', end='')

        print(f"Begin Evaluation Client: {conn_client.id}")


        for iteration in range(num_test_iterations):
            conn_client.get_front_activations()
            conn_client.forward_center()
            conn_client.send_center_activations()

        
        lock.acquire()
        clients_done = clients_done + 1
        lock.release() 
  

        while (signal[i] != 1):
            time.sleep(0.2)

        send_message = "Begin next epoch"
        # print(send_message.encode("utf-8"))
        conn_client.conn.sendall(send_message.encode())
        




if __name__=="__main__":



    print("started")

    parser = argparse.ArgumentParser(description='Example program that accepts command line arguments.')
    parser.add_argument('--global_epochs', default = 5)
    parser.add_argument('--num_clients',default = 2)
    parser.add_argument('--port',default = 9092)
    args = parser.parse_args()

    MODEL = importlib.import_module(f'models.resnet18')
    HEADERSIZE = 20
    GLOBAL_EPOCHS = int(args.global_epochs)
    TOTAL_CLIENTS= int(args.num_clients)
    models=[0] * TOTAL_CLIENTS
    threads=[]
    clients_done = 0
    signal = [0] * GLOBAL_EPOCHS   #global epoch 
    LR = 0.001
    BATCH_SIZE = 32
    PORT = int(args.port)

    
    lock = threading.Lock()
    serverSocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    serverSocket.bind(("127.0.0.1",PORT))
    serverSocket.listen()
    print("listening")

    for i in range(TOTAL_CLIENTS):
        c,add=serverSocket.accept()
        t=threading.Thread(target=handle_client,args=(c,i,lock))
        t.start()
        threads.append(t)

    
    #MAIN THREAD (ALSO HANDLES AGGREGATION)

    for i in range(GLOBAL_EPOCHS):

        while(clients_done != TOTAL_CLIENTS):
            time.sleep(0.2)

        print("Main thread", clients_done)


        lock.acquire() 
        clients_done = 0
        lock.release()
        
        params = []
        for j in range(TOTAL_CLIENTS):
            params.append(copy.deepcopy(models[j].state_dict()))

        w_glob = merge_weights(params)

        for j in range(TOTAL_CLIENTS):
            models[j].load_state_dict(w_glob)

        signal[i] = 1


    for t in threads:
            t.join()
           
    print("Everything done")
    
    

