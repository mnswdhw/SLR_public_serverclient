import socket
import torch 
import numpy as np 
from io import BytesIO
import torch.optim as optim 
import torch.nn.functional as F
import importlib
from utils import datasets, dataset_settings
from torch.utils.data import DataLoader, Dataset




MODEL = importlib.import_module(f'models.resnet18')
HEADERSIZE = 20
GLOBAL_EPOCHS = 5
TOTAL_CLIENTS=2
LR = 0.001
BATCH_SIZE = 32
PORT = 9093


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    
            


train_full_dataset, test_full_dataset, input_channels = datasets.load_full_dataset("cifar10_tl", "data", TOTAL_CLIENTS, 100)  #50 is the data per client
dict_users = dataset_iid(train_full_dataset, TOTAL_CLIENTS)
dict_users_test = dataset_iid(test_full_dataset, TOTAL_CLIENTS)




class Client():
    def __init__(self):
        self.id = None
        self.losses = []
        self.train_dataset = None
        self.test_dataset = None
        self.train_DataLoader = None
        self.test_DataLoader = None
        self.socket = None
        self.server_socket = None
        self.train_batch_size = None
        self.test_batch_size = None
        self.iterator = None
        self.activations1 = None
        self.remote_activations1 = None
        self.outputs = None
        self.loss = None
        self.criterion = None
        self.data = None
        self.targets = None
        self.n_correct = 0
        self.n_samples = 0
        self.front_optimizer = None
        self.back_optimizer = None
        self.test_acc = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def initialize(self):

        self.front_model = MODEL.front(pretrained = True)
        self.back_model = MODEL.back(pretrained = True)
        self.front_optimizer = optim.Adam(self.front_model.parameters(), lr=LR)
        self.back_optimizer = optim.Adam(self.back_model.parameters(), lr=LR)
        self.train_dl = DataLoader(DatasetSplit(train_full_dataset, dict_users[self.id]), batch_size = BATCH_SIZE, shuffle = True)
        self.test_dl = DataLoader(DatasetSplit(test_full_dataset, dict_users_test[self.id]), batch_size = BATCH_SIZE, shuffle = True)
        print("HELLLLLO", len(self.train_dl))
        print("HELLLLLO", len(self.test_dl))


    def connect(self):

        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect(("127.0.0.1",PORT))
        self.id = int(self.conn.recv(1024).decode())

    def backward_back(self):
        self.loss.backward()
        

    def backward_front(self):
        self.activations_front.backward(self.remote_activations_front.grad)


    def calculate_loss(self):
        self.criterion = F.cross_entropy
        self.loss = self.criterion(self.outputs, self.targets)
        # print(self.outputs.shape)
        # print(target.shape)
        # self.loss = self.criterion(self.outputs, target)


    def calculate_test_acc(self):
        with torch.no_grad():
            _, self.predicted = torch.max(self.outputs.data, 1)
            self.n_correct = (self.predicted == self.targets).sum().item()
            self.n_samples = self.targets.size(0)
            return 100.0 * self.n_correct/self.n_samples



    def calculate_train_acc(self):
        with torch.no_grad():
            _, self.predicted = torch.max(self.outputs.data, 1)
            self.n_correct = (self.predicted == self.targets).sum().item()
            self.n_samples = self.targets.size(0)
            return 100.0 * self.n_correct/self.n_samples



    def create_DataLoader(self, train_batch_size, test_batch_size):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_DataLoader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                batch_size=self.train_batch_size,
                                                shuffle=True)
        self.test_DataLoader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                batch_size=self.test_batch_size,
                                                shuffle=True)


    def forward_back(self):
        self.back_model.to(self.device)
        self.outputs = self.back_model(self.activations_center)


    def forward_front(self):
        self.data, self.targets = next(self.iterator)
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)
        self.front_model.to(self.device)
        self.activations_front = self.front_model(self.data)
        self.remote_activations_front = self.activations_front.detach().requires_grad_(True)


    def get_remote_activations_front_grads(self):
        self.remote_activations_front.grad = self.receive_tensor()


    def get_activations_center(self):
        self.activations_center = self.receive_tensor()
        self.activations_center.requires_grad = True  #as after receiving tensor requires grad is false
        # print("naruto", self.activations_center.requires_grad)

    def send_remote_activations_front(self):
        self.send_tensor(self.remote_activations_front)
    

    def send_activations_center_grads(self):
        # print(self.activations_center.grad)
        self.send_tensor(self.activations_center.grad)


    def step_front(self):
        self.front_optimizer.step()
        

    def step_back(self):
        self.back_optimizer.step()

    def zero_grad_front(self):
        self.front_optimizer.zero_grad()
        

    def zero_grad_back(self):
        self.back_optimizer.zero_grad()


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
        print("frame received")
        return torch.from_numpy(arr)


