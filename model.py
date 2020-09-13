#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(3, 10, kernel_size=5, padding=2),
                                          torch.nn.ReLU(), torch.nn.BatchNorm2d(num_features=10),
                                          torch.nn.MaxPool2d(2, 2))
        
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(10, 20, kernel_size=3, padding=1),
                                          torch.nn.Dropout2d(p=0.2), torch.nn.ReLU(),
                                          torch.nn.BatchNorm2d(num_features=20), torch.nn.MaxPool2d(2, 2))

        self.encoder = torch.nn.Sequential(self.layer1, self.layer2)

        self.decoder = torch.nn.Sequential(torch.nn.Linear(56*56*20, 6000), torch.nn.ReLU(),
                                           torch.nn.Linear(6000, 500), torch.nn.ReLU(), torch.nn.Linear(500, 15))

    def forward(self, x):
        x = self.encoder(x)

        #flatten heatmap before utilizing dense layers
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        out = self.decoder(x)
        return out

    
class LRCN(torch.nn.Module):
    def __init__(self):
        super(LRCN, self).__init__()
        self.cnn = CNN()
        #batch first: data formatted in (batch, seq, feature)
        self.rnn = torch.nn.LSTM(input_size=15, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(64, 1)

    def forward(self, x):
        heatmaps = []
        for seq in x:
            heatmaps.append(self.cnn.forward(seq))
        heatmaps = torch.stack(heatmaps)
        out, (_, _) = self.rnn(heatmaps)
        out = self.linear(out)
        return out[:,-1,:] 

