import numpy as np
import torch as T
from emnist import extract_training_samples
from emnist import extract_test_samples
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Flatten
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


class Character_Recognition():
    #Character_Recognition is a class that contains functions to:
    #  Train a given CNN model
    #  Test a trained CNN model
    #  Decode the EMNIST balanced dataseet labels into characters
    #  Performe inference using a trained CNN model

    def train(cnn, num_epochs, loaders):    
        cnn.train()
        total_step = len(loaders['train'])

        #Defining the Adam optimizer with learning rate 0.001
        optimizer = Adam(cnn.parameters(), lr=0.001)
        
        #Defining loss function
        criterion = CrossEntropyLoss()
    
        #Uses gpu  to train if available. If it is not, uses cpu
        if T.cuda.is_available():
            T.cuda.empty_cache()
            cnn = cnn.cuda()
            criterion = criterion.cuda()

        #Empty list to store training losses
        train_losses = []

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(loaders['train']):
                #Adds padding to [left, right, top, bot] of the training images
                images = F.pad(images, (1, 1, 1, 1))
                images_batch = Variable(images).float()
                labels_batch = Variable(labels)

                #Uses gpu to train if available. If it is not, uses cpu   
                if T.cuda.is_available():
                    images_batch = images_batch.cuda()
                    labels_batch = labels_batch.cuda()

                #Apply input batch at the cnn
                output = cnn(images_batch)  
                #Compute the loss using the cnn output and the target output
                loss = criterion(output, labels_batch) 
                #Clear gradients for this training step   
                optimizer.zero_grad()           
                #Backpropagation, compute gradients 
                loss.backward()    
                #Apply gradients             
                optimizer.step()   
                #Save loss values 
                train_losses.append(loss.cpu().detach().numpy())          

                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                    pass
            pass
        
        plt.plot(train_losses, label='Training loss')
        plt.show()
        pass

    def test(test_images, test_labels):

        #Loads a trained cnn model    
        model = CNN()
        model.load_state_dict(T.load("trained_cnn.pt"))
        
        #Add padding to images
        test_images = F.pad(test_images, (1, 1, 1, 1))

        #Apply test images into the cnn
        #Uses gpu to test if available. If it is not, uses cpu
        if T.cuda.is_available():
            T.cuda.empty_cache()
            model = model.cuda()
            with T.no_grad():
                output = model((test_images.float()).cuda())
        else:
            with T.no_grad():
                output = model((test_images.float()).cpu().detach())

        #Gets classification prediction
        softmax = T.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)

        #Accuracy on testing set
        print("CNN tested with EMNIST balanced test dataset and presents {}% of accuracy".format(accuracy_score(test_labels, predictions)*100))



    def decode(label):
        #Decode the letters for the balanced EMNIST dataset---------------------------------
        merged_letters = ["c", "i", "j", "k", "l", "m", "o", "p", "s", "u", "v", "w", "x", "y", "z"]
        uppercase_values = {i - 55: chr(i) for i in range(ord("A"), ord("A") + 26)}

        lowercase_values = dict()
        aux = 97
        for i in range(ord("a"), ord("a") + 26):
            if(chr(i) in merged_letters):
                lowercase_values[ord(chr(i)) - 87] = chr(i)
            else:
                lowercase_values[aux - 61] = chr(i)
                aux += 1

        if(type(label) == int):
            if((label >= 0) and (label<10)):
                decoded_label = label
            elif((label > 9) and (label<36)):
                decoded_label = uppercase_values[label]
            elif((label > 35) and (label<48)):
                decoded_label = lowercase_values[label]
            else:
                decoded_label = 'error'
                print("CNN ERROR: Label out of range. It should be between 0 and 47, but is {}".format(label))
        else:
            decoded_label = 'error'
            print("CNN ERROR: Wrong label type. It should be an int, but is {}".format(type(label)))
            
    
        return(decoded_label)
    

    def classify(input):
        #Gets an image as input and return character---------------------------

        #Tests to check input dimensions
        if((input.shape[-1] != 28) or (input.shape[-2] != 28)):
            print("CNN ERROR: Image dimensions wrong. It should be 28x28, it is {}x{}".format(input.shape[-2], input.shape[-1]))
            exit()

        if(input.shape == T.Size([1, 28, 28])):
            input = input.view(1, 1, 28, 28)

        if(input.shape == (28,28)):
            input = T.from_numpy((input).reshape(1, 1, 28, 28))

        #Load trained cnn model
        model = CNN()
        model.load_state_dict(T.load("trained_cnn.pt"))

        #Gets cnn prediction
        model.eval()
        output = model(input.float())
        prediction = int(T.max(output.data, 1)[1].numpy())

        return prediction



class CNN(Module):   
    #CNN class defines the network architecture-----------------------------
    
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = Sequential(
            # Defining 2D convolution layer 1
            Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=1,),
            ReLU(),
            BatchNorm2d(8),
            MaxPool2d(kernel_size=2),
            
            # Defining 2D convolution layer 2
            Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1,),
            ReLU(),
            BatchNorm2d(16),
            MaxPool2d(kernel_size=2),

            # Defining 2D convolution layer 3
            Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1,),
            ReLU(),
            
        )
        
        self.linear_layers = Sequential(
            Linear(24 * 7 * 7, 128),
            ReLU(),
            Dropout(0.2),
            Linear(128, 47),
                     )


    # Defining the forward pass    
    def forward(self, x): 
        x = self.cnn_layers(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)  
        x = self.linear_layers(x)
        return x   



if __name__ == '__main__': 

    #Load balanced EMNIST images and labels 
    train_images_original, train_labels_original = extract_training_samples('balanced')
    test_images_original, test_labels_original = extract_test_samples('balanced')

    #Normalizes train and test images, reshapes to (number of images, number of channels, dim1, dim2) and creates tensor
    train_images = T.from_numpy((train_images_original.copy()/255.0).reshape(112800, 1, 28, 28))
    test_images  = T.from_numpy((test_images_original.copy()/255.0).reshape(18800, 1, 28, 28))

    #Convert labels into tensors
    train_labels = T.from_numpy((train_labels_original.copy().astype(int)))
    test_labels  = T.from_numpy((test_labels_original.copy().astype(int)))

    #Creates training and testing datasets
    EMNIST_train = TensorDataset(train_images,train_labels) 
    EMNIST_test  = TensorDataset(test_images,test_labels) 

    #Creates dataloaders to train and test datasets with batch size and shuffle property
    loaders = {'train' : DataLoader(EMNIST_train, batch_size=700, shuffle=True),
               'test'  : DataLoader(EMNIST_test,  batch_size=125,  shuffle=True)}

   
    #Defining the CNN model
    cnn = CNN()
    
    #Defining the number of epochs
    n_epochs = 78
    
    #Training and saving the CNN
    Character_Recognition.train(cnn, n_epochs, loaders)
    T.save(cnn.state_dict(), "./trained_cnn.pt")

    #Testing the trained CNN
    Character_Recognition.test(test_images, test_labels)



    
    

        
    

