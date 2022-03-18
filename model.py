
"""
Description: model file to train beat tracking model
"""


import torch.nn as nn
import torch



class Beat_tracking(nn.Module):
    def __init__(
            self,
            channels=16,
            dropout=0.1):
        """

        Keyword Arguments:
            Input dimensions (default: {(3000, 81)})
            Output dimensions (default: {3000})
            channels {int} -- Convolution channels (default: {16})
            dropout {float} -- Network connection dropout probability.
                               (default: {0.1})
        """
        super(Beat_tracking, self).__init__()

        self.conv1 = nn.Conv2d(1, channels, (3, 3), padding=(1, 0))
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool2d((1, 3))

        self.conv2 = nn.Conv2d(channels, channels, (3, 3), padding=(1, 0))
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool2d((1, 3))

        self.conv3 = nn.Conv2d(channels, channels, (1, 8))
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(dropout)
       
        self.out = nn.Conv1d(channels, 2 , 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)
        y = self.pool2(y)

        y = self.conv3(y)
        y = self.elu3(y)
       

        y = y.view(-1, y.shape[1], y.shape[2])
        
        y =self.out(y)
        y = self.sigmoid(y)

        return y
