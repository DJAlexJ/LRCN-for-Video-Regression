import torch
import preprocessing as prep
from loading_data import load_data

from config import TRAINING_PATH, PREDICTION_PATH, TEST_TRAILER_NAME, PATH_MODEL_CHECKPOINT


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', torch.nn.ReLU()],
        ['leaky_relu', torch.nn.LeakyReLU(negative_slope=0.01)],
        ['selu', nn.SELU()]
    ])[activation]

def optimizer_choice(optimizer, parameters, lr=3e-4):
    return torch.nn.ModuleDict([
        ['adam', torch.optim.Adam(parameters, lr=lr)],
        ['rmsprop', torch.optim.RMSprop(parameters, lr=lr)],
        ['sgd', torch.optim.SGD(parameters, lr=lr)]
    ])[optimizer]

def loss_choice(loss):
    return torch.nn.ModuleDict([
        ['mse', torch.nn.MSELoss()],
        ['mae', torch.nn.L1Loss()],
        ['smooth_mae', torch.nn.SmoothL1Loss()]
    ])[loss]


class CNN(torch.nn.Module):
    def __init__(self, channels1=10, channels2=20, embedding_size=15, activation='relu'):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(3, channels1, kernel_size=5, padding=2),
                                          activation_func(activation), torch.nn.BatchNorm2d(num_features=channels1),
                                          torch.nn.MaxPool2d(2, 2))
        
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(channels1, channels2, kernel_size=3, padding=1),
                                          torch.nn.Dropout2d(p=0.2), activation_func(activation),
                                          torch.nn.BatchNorm2d(num_features=channels2), torch.nn.MaxPool2d(2, 2))

        self.encoder = torch.nn.Sequential(self.layer1, self.layer2)

        self.decoder = torch.nn.Sequential(torch.nn.Linear(56*56*channels2, 6000), activation_func(activation),
                                           torch.nn.Linear(6000, 500), activation_func(activation), torch.nn.Linear(500, embedding_size))

    def forward(self, x):
        x = self.encoder(x)

        #flatten heatmap before utilizing dense layers
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        out = self.decoder(x)
        return out

    
class LRCN(torch.nn.Module):
    def __init__(self, channels1=10, channels2=20, embedding_size=15, LSTM_size=64, LSTM_layers=1, activation='relu'):
        super(LRCN, self).__init__()
        self.cnn = CNN(channels1, channels2, embedding_size, activation)
        #batch first: data formatted in (batch, seq, feature)
        self.rnn = torch.nn.LSTM(input_size=embedding_size, hidden_size=LSTM_size, num_layers=LSTM_layers, batch_first=True)
        self.linear = torch.nn.Linear(LSTM_size, 1)

    def forward(self, x):
        heatmaps = []
        for seq in x:
            heatmaps.append(self.cnn.forward(seq))
        heatmaps = torch.stack(heatmaps)
        out, (_, _) = self.rnn(heatmaps)
        out = self.linear(out)
        return out[:,-1,:]
    
    def train(self, dir_names, X_test, y_test, lr=3e-4, loss_name='mse', optimizer_name='adam', n_epoch=5, batch_size=10, device='cpu', use_checkpoint=False, use_tensorb=False, verbose=False):
        
        #Activating tensorboard
#         if use_tensorb:
#             tb = SummaryWriter()

        optimizer = optimizer_choice(optimizer_name, self.parameters(), lr)
        loss = loss_choice(loss_name)
      
        dir_names = list(filter(lambda x: os.path.isdir(f"{TRAINING_PATH}/{x}"), dir_names)) #Filtering waste files
        random.shuffle(dir_names)

        train_loss_history = []
        test_loss_history = []

        learning_dir_names = dir_names.copy()
        #Training model
        for epoch in range(n_epoch):
            dir_names = learning_dir_names.copy()
            train_loss = 0
            for i in range(0, len(learning_dir_names), batch_size):
                
                optimizer.zero_grad()
                X_batch, y_batch, dir_names = load_data(dir_names, verbose, batch_size=batch_size)  

                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float()

                preds = self.forward(X_batch).view(y_batch.size()[0])
                loss_value = loss(preds, y_batch)
                loss_value.backward()

                train_loss += loss_value.data.cpu()
                optimizer.step()

            train_loss_history.append(train_loss)

            with torch.no_grad():
                test_preds = model.forward(X_test).view(y_test.size()[0])
                test_loss_history.append(loss(test_preds, y_test).data.cpu())

            #Saving checkpoint
            if use_checkpoint:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        loss_name: test_mse_history[-1],
                        }, PATH_MODEL_CHECKPOINT)

#             if use_tensorb:
#                 tb.add_scalar(loss_name, test_loss_history[-1], epoch)

            print(f"{epoch+1}: {loss_name} = {test_loss_history[-1]}")

#         if use_tensorb:
#             tb.close()

        return [train_loss_history, test_loss_history]

    def predict(self, movie_name, verbose=False)
    
        prep.preprocess(movie_name, train=False, n_subclips=1, verbose=verbose)
        name = movie_name.rsplit('.', 1)[0]+'_0' #Getting appropriate name
        image = load_data([name], train=False, verbose=verbose, batch_size=1)
        prediction = model.forward(image).detach().unsqueeze(-1).item()
        
        return prediction
