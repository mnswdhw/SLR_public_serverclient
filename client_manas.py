import torch 
import importlib
from client_class import Client
from math import ceil
import argparse








if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Example program that accepts command line arguments.')
    parser.add_argument('--global_epochs', default = 5)
    parser.add_argument('--num_clients',default = 2)
    parser.add_argument('--port',default = 9092)
    args = parser.parse_args()


    MODEL = importlib.import_module(f'models.resnet18')
    HEADERSIZE = 20
    GLOBAL_EPOCHS = int(args.global_epochs)
    TOTAL_CLIENTS= int(args.num_clients)
    LR = 0.001
    BATCH_SIZE = 32

    client = Client()
    client.connect()
    client.initialize()
    #get id 
    # print("The client id is ", client.id)
    
    num_iterations = ceil(len(client.train_dl.dataset)/BATCH_SIZE)

    num_test_iterations = ceil(len(client.test_dl.dataset)/BATCH_SIZE)
    test_accuracy = [0] * GLOBAL_EPOCHS
    print("Test iterations", num_test_iterations)
    print(len(client.test_dl.dataset))
    print("Train iterations", num_test_iterations)



    for epoch in range(GLOBAL_EPOCHS):

        client.iterator = iter(client.train_dl)
        

        for iteration in range(num_iterations):

            print(f'\rEpoch: {epoch+1}, Iteration: {iteration+1}/{num_iterations}', end='')

            client.forward_front()
            client.send_remote_activations_front()
            client.get_activations_center()
            client.forward_back()
            client.calculate_loss()
            client.backward_back()
            client.send_activations_center_grads()
            # client.get_remote_activations_front_grads()
            # client.backward_front()

            # client.step_front()
            client.step_back()
            # client.zero_grad_front()
            client.zero_grad_back()
  

        print("Begin Evaluation")
        client.iterator = iter(client.test_dl)
        client.test_acc.append(0)

        for iteration in range(num_test_iterations):

            client.forward_front()
            client.send_remote_activations_front()
            client.get_activations_center()
            client.forward_back()
            client.test_acc[-1] += client.calculate_test_acc()

        client.test_acc[-1] /= num_test_iterations

        print(f"Test Accuracy Epoch {epoch}: {client.test_acc[-1]}")

            
        msg=client.conn.recv(1024).decode() 
        print(f"recieved message: {msg}") 
            



    

    

