import torch
import numpy as np

device = torch.device("cpu")
class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=6,
            hidden_size=90,
            num_layers=4,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )  
        self.softmax = torch.nn.Softmax(dim=1) 
        self.fc = torch.nn.Linear(90,2)

        model = torch.load('results/pretrained-models/zv_lstm_model.tar')
        my_dict = self.state_dict()
        for key,value in my_dict.items(): 
            my_dict[key]=model[key]
        self.load_state_dict(my_dict)
        self.eval()
    def forward(self, x, h=None, mode="train"):
        x = torch.FloatTensor(x).view((1,-1, 6))
        if h is None:       ##initalize the hidden output if not already initialized
            h_n = x.data.new(4, x.size(0), 90).normal_(0, 0.1)
            h_c = x.data.new(4, x.size(0), 90).normal_(0, 0.1)  
        else:              
            h_n, h_c = h  
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(x, (h_n, h_c))    #uncomment for LSTM
        output = self.softmax(self.fc(r_out[0,:,:]))    #grab the whole sequence at test time
        zv_lstm = torch.max(output.cpu().data,1)[1].numpy()
        prob = torch.max(output.cpu().data,1)[0].numpy()
        zv_lstm[np.where(prob<=0.85)]=0
        return zv_lstm #output, (h_n, h_c)
    
