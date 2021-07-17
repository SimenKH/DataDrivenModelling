# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:37:33 2021

@author: simen
"""


import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from math import sqrt
import sys
import xlsxwriter
import numpy as np
import GPUtil as GPU
import math
from threading import Thread
import time
# class Monitor(Thread):
#     def __init__(self, delay):
#         super(Monitor, self).__init__()
#         self.stopped = False
#         self.delay = delay # Time between calls to GPUtil
#         self.start()

#     def run(self):
#         while not self.stopped:
#             GPU.showUtilization()
#             time.sleep(self.delay)

#     def stop(self):
#         self.stopped = True
class Net(torch.nn.Module):
    """
    PyTorch offers several ways to construct neural networks.
    Here we choose to implement the network as a Module class.
    This gives us full control over the construction and clarifies our intentions.
    """
    
    def __init__(self, layers):
        """
        Constructor of neural network
        :param layers: list of layer widths. Note that len(layers) = network depth + 1 since we incl. the input layer.
        """
        super().__init__()

        self.device = 'cuda' #if torch.cuda.is_available() else 'cpu'

        assert len(layers) >= 2, "At least two layers are required (incl. input and output layer)"
        self.layers = layers

        # Fully connected linear layers
        linear_layers = []

        for i in range(len(self.layers) - 1):
            n_in = self.layers[i]
            n_out = self.layers[i+1]
            layer = torch.nn.Linear(n_in, n_out)

            # Initialize weights and biases
            a = 1 if i == 0 else 2
            layer.weight.data = torch.randn((n_out, n_in)) * sqrt(a / n_in)
            layer.bias.data = torch.zeros(n_out)
            
            # Add to list
            linear_layers.append(layer)
        
        # Modules/layers must be registered to enable saving of model
        self.linear_layers = torch.nn.ModuleList(linear_layers)  

        # Non-linearity (e.g. ReLU, ELU, or SELU)
        self.act = torch.nn.ReLU(inplace=False)

    def forward(self, input):
        """
        Forward pass to evaluate network for input values
        :param input: tensor assumed to be of size (batch_size, n_inputs)
        :return: output tensor
        """
        x = input
        for l in self.linear_layers[:-1]:
            x = l(x)
            x = self.act(x)

        output_layer = self.linear_layers[-1]
        return output_layer(x)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str):
        """
        Save model state
        :param path: Path to save model state
        :return: None
        """
        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)

    def load(self, path: str):
        """
        Load model state from file
        :param path: Path to saved model state
        :return: None
        """
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.load_state_dict(checkpoint['model_state_dict'])
        
def train(
        net: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        lr: float,
        l2_reg: float, MSElist, gpu
) -> torch.nn.Module:
    """
    Train model using mini-batch SGD
    After each epoch, we evaluate the model on validation data
    :param net: initialized neural network
    :param train_loader: DataLoader containing training set
    :param n_epochs: number of epochs to train
    :param lr: learning rate (default: 0.001)
    :param l2_reg: L2 regularization factor (default: 0)
    :return: torch.nn.Module: trained model.
    """

    # Define loss and optimizer
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)#, lr_decay=0.001*lr*(1/n_epochs))

    # Train Network
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            # Zero the parameter gradients (from last iteration)
            optimizer.zero_grad()

            # Forward propagation
            outputs = net(inputs)
            
            # Compute cost function
            batch_mse = criterion(outputs, labels)
            
            reg_loss = 0
            for param in net.parameters():
                reg_loss += param.pow(2).sum()

            cost = batch_mse + l2_reg * reg_loss

            # Backward propagation to compute gradient
            cost.backward()
            
            # Update parameters using gradient
            optimizer.step()
            #print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
        
        # Evaluate model on validation data
        mse_val = 0
        for inputs, labels in val_loader:
            mse_val += torch.sum(torch.pow(labels - net(inputs), 2)).item()
        mse_val /= len(val_loader.dataset)
        MSElist.append(mse_val)
        #print(f'Epoch: {epoch + 1}: Val MSE: {mse_val}')
        
    return net

def main(hidden_layers,epochs,learning_rate,result_filename,console_log_filename,time_spent_filename,output,inputset):
    
 
    # monitor = Monitor(5)
    GPUs = GPU.getGPUs()
    gpu = GPUs[0]
    t0=time.time()
    
    
    random_seed =13371337 
    randSeed2=12369784
    # This seed is also used in the pandas sample() method below
    torch.manual_seed(random_seed)
    df = pd.read_csv(r"C:\Users\simen\OneDrive\Skrivebord\MasterOppgave\Engine\SeptDec2020-ver3.csv", dtype=np.float64)
    for col in df:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print('Sizze of dataset', df.shape)
    # Test set (this is the period for which we must estimate QTOT)
    test_set =df.sample(frac=0.1, replace=False, random_state=randSeed2)

    # Make a copy of the dataset and remove the test data
    train_val_set = df.copy().drop(test_set.index)

    # Sample validation data without replacement (10%)
    val_set = train_val_set.sample(frac=0.1, replace=False, random_state=random_seed)

    # The remaining data is used for training (90%)
    train_set = train_val_set.copy().drop(val_set.index)

    # Check that the numbers add up
    n_points = len(train_set) + len(val_set) + len(test_set)
    print(f'{len(df)} = {len(train_set)} + {len(val_set)} + {len(test_set)} = {n_points}')
    INPUT_COLS = inputset	#'Africa.571_TT_114','Africa.871_XI_10151',	'Africa.871_XI_10207','Africa.871_XI_10259','Africa.871_XI_10302','Africa.871_XI_10303','Africa.871_XI_10304','Africa.871_XI_10306','Africa.871_XI_10312','Africa.871_XI_10315','Africa.871_XI_10363','Africa.871_XI_10409']
    OUTPUT_COLS = output
    
    # Get input and output tensors and convert them to torch tensors
    x_train = torch.from_numpy(train_set[INPUT_COLS].values).to(torch.float)
    y_train = torch.from_numpy(train_set[OUTPUT_COLS].values).to(torch.float)

    x_val = torch.from_numpy(val_set[INPUT_COLS].values).to(torch.float)
    y_val = torch.from_numpy(val_set[OUTPUT_COLS].values).to(torch.float)

    # Create dataset loaders
    # Here we specify the batch size and if the data should be shuffled
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8192, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_set), shuffle=False)
    
    layers_constructor = [len(INPUT_COLS)]+hidden_layers+[len(OUTPUT_COLS)]
    layers = layers_constructor
    net = Net(layers)

    print(f'Layers: {layers}')
    print(f'Number of model parameters: {net.get_num_parameters()}')
    
    n_epochs = epochs #orig 12000
    lr =  1
    l2_reg =  0.007  # 10
   
    
    MSElist=[]
  
    net = train(net, train_loader, val_loader, n_epochs, lr, l2_reg,MSElist,gpu)
   
    # with open('MSEovertime.txt','w') as f:
    #     sys.stdout=f
    #     epoke=1
    #     for element in MSElist:
    #         print('Epoch:',epoke, '     Val MSE: ' ,element, '\n \n')
    #         epoke+=1
        
    
    workbook = xlsxwriter.Workbook(result_filename)
    worksheet = workbook.add_worksheet()
    row=0
    for element in MSElist:
        worksheet.write(row,0,element)
        row+=1
    workbook.close()
    # monitor.stop()
    t1=time.time()-t0
    
    print('Time for entire script to run:', t1)
   
    #Get input and output as torch tensors
    x_test = torch.from_numpy(test_set[INPUT_COLS].values).to(torch.float)
    

    
    y_test = torch.from_numpy(test_set[OUTPUT_COLS].values).to(torch.float)

    Make prediction
    pred_test = net(x_test)

    Compute MSE, MAE and MAPE on test data
    print('Error on test data')
    
    mse_test = torch.mean(torch.pow(pred_test - y_test, 2))
    print(f'MSE: {mse_test.item()}')
    
    mae_test = torch.mean(torch.abs(pred_test - y_test))
    print(f'MAE: {mae_test.item()}')
    
    mape_test = 100*torch.mean(torch.abs(torch.div(pred_test - y_test, y_test)))
    print(f'MAPE: {mape_test.item()} %')
    
def multi_run(mode):
    #constants
    epochs=12000
    learning_rate=0.7
    #rectangular mode
    if (mode==1):
       modus="Rectangular"
       hidden_layers=[]
       max_depth=20
       max_width=100
       depthlist=range(2,max_depth)
       widthlist=np.arange(10,max_width+10,10).tolist()
       for depth in depthlist:
           for width in widthlist:
               hidden_layers=[width]*depth
               result_filename="results_using_"+modus+"neuralnet_with_depth_"+str(depth)+"_and width_"+str(width)+".xlsx"
               console_log_filename="console_log_using_"+modus+"neuralnet_with_depth_"+str(depth)+"_and width_"+str(width)+".txt"
               time_spent_filename="time_spent_using_"+modus+"neuralnet_with_depth_"+str(depth)+"_and width_"+str(width)+".txt"
               main(hidden_layers,epochs,learning_rate,result_filename,console_log_filename,time_spent_filename)
    
    
    if (mode==2):
        modus="Cone"
        hidden_layers=[]
        max_depth=100
        max_width=500
        depthlist=range(2,max_depth)
        widthlist=np.arange(10,max_width+10,10).tolist()
        ratio_between_width_of_first_and_last_layer=30
        for depth in depthlist:
            for width in widthlist:
                first_layer=width
                layer_width_change=math.floor((first_layer/ratio_between_width_of_first_and_last_layer)/depth)
                hidden_layers=[None]*depth
                
                for i in range(len(hidden_layers)):
                    if (i==0):
                        hidden_layers[i]=first_layer
                    else:
                        hidden_layers[i]=hidden_layers[i-1]-layer_width_change
                result_filename="results_using_"+modus+"_neuralnet_with_depth_"+str(depth)+"start_width="+str(hidden_layers[0])+"_last_layer_"+str(hidden_layers[-1])+".xlsx"
                console_log_filename="console_log_using_"+modus+"_neuralnet_with_depth_"+str(depth)+"start_width="+str(hidden_layers[0])+"_last_layer_"+str(hidden_layers[-1])+".txt"
                time_spent_filename="time_spent_using_"+modus+"_neuralnet_with_depth_"+str(depth)+"start_width="+str(hidden_layers[0])+"_last_layer_"+str(hidden_layers[-1])+".txt"
                main(hidden_layers,epochs,learning_rate,result_filename,console_log_filename,time_spent_filename)
                        
    
def GrowingApproach():
    output=['Africa.871_XI_10207']
    inputs=[['Africa.601_XI_10114','Africa.601_UA_10176','Africa.601_TI_10199','Africa.601_TI_10200','Africa.601_TI_10179','Africa.601_TI_10179','Africa.601_TI_10178'],['Africa.601_XI_10114','Africa.601_UA_10176','Africa.601_TI_10199','Africa.601_TI_10200','Africa.601_TI_10179','Africa.601_TI_10179','Africa.601_TI_10178','Africa.601_PI_10177','Africa.601_PT_10163'],['Africa.601_XI_10114','Africa.601_UA_10176','Africa.601_TI_10199','Africa.601_TI_10200','Africa.601_TI_10179','Africa.601_TI_10179','Africa.601_TI_10178','Africa.601_PI_10177','Africa.601_PT_10163','Africa.601_PI_10170','601_TI_10172'],['Africa.601_XI_10114','Africa.601_UA_10176','Africa.601_TI_10199','Africa.601_TI_10200','Africa.601_TI_10179','Africa.601_TI_10179','Africa.601_TI_10178','Africa.601_PI_10177','Africa.601_PT_10163','Africa.601_PI_10170','601_TI_10172','Africa.601_TT_10189','Africa.601_TT_10188','Africa.601_TT_10187','Africa.601_TT_10186','Africa.601_TT_10185','Africa.601_TT_10184','Africa.601_TT_10183','Africa.601_TT_10182','Africa.601_TT_10181']]
    
    #inputs=[['Africa.601_XI_10114'],['Africa.601_XI_10114','Africa.601_UA_10176','Africa.601_TI_10199','Africa.601_TI_10200'],['Africa.601_XI_10114','Africa.601_UA_10176','Africa.601_TI_10199','Africa.601_TI_10200','Africa.601_TI_10179','Africa.601_TI_10179','Africa.601_TI_10178'],['Africa.601_XI_10114','Africa.601_UA_10176','Africa.601_TI_10199','Africa.601_TI_10200','Africa.601_TI_10179','Africa.601_TI_10179','Africa.601_TI_10178','Africa.601_PI_10177','Africa.601_PT_10163'],['Africa.601_XI_10114','Africa.601_UA_10176','Africa.601_TI_10199','Africa.601_TI_10200','Africa.601_TI_10179','Africa.601_TI_10179','Africa.601_TI_10178','Africa.601_PI_10177','Africa.601_PT_10163','Africa.601_PI_10170','601_TI_10172'],['Africa.601_XI_10114','Africa.601_UA_10176','Africa.601_TI_10199','Africa.601_TI_10200','Africa.601_TI_10179','Africa.601_TI_10179','Africa.601_TI_10178','Africa.601_PI_10177','Africa.601_PT_10163','Africa.601_PI_10170','601_TI_10172','Africa.601_TT_10189','Africa.601_TT_10188','Africa.601_TT_10187','Africa.601_TT_10186','Africa.601_TT_10185','Africa.601_TT_10184','Africa.601_TT_10183','Africa.601_TT_10182','Africa.601_TT_10181']]
    teller=0
    for inputset in inputs:
        # if inputset != inputs[-1]:
        #     continue
        epochs=6000
        learning_rate=0.7
        hidden_layers=[500,250,125,75,10]
        result_filename="CONEresults_using_input_set_"+str(teller)+".xlsx"
        console_log_filename="CONEconsole_log_using_input_set"+str(teller)+".txt"
        time_spent_filename="CONEtime_spent_using_input_set"+str(teller)+".txt"
        main(hidden_layers,epochs,learning_rate,result_filename,console_log_filename,time_spent_filename,output,inputset)
        teller+=1
    
    
