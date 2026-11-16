import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchinfo import summary
import math



class DConformerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.DilatedConv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size = 3, padding = 'same', dilation = 2),
            nn.BatchNorm1d(16),
            nn.ReLU()
        ) 
        self.Maxpool1 = nn.MaxPool1d(kernel_size = 2 , stride = 2) # 1024
        
        self.DilatedConv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size = 3, padding = 'same', dilation = 2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        ) # 1024
        self.Maxpool2 = nn.MaxPool1d(kernel_size = 2 , stride = 2) # 512
        
        self.DilatedConv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size = 3, padding = 'same', dilation = 2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        ) 
        self.Maxpool3 = nn.MaxPool1d(kernel_size = 2 , stride = 2) # 256
        
        self.DilatedConv4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size = 3, padding = 'same', dilation =2 ), # 
            nn.BatchNorm1d(128),
            nn.ReLU()
        ) 
        self.Maxpool4 = nn.MaxPool1d(kernel_size = 2 , stride = 2) # 128
        
        self.DilatedConv5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size = 3, padding = 'same', dilation = 2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        ) 
        self.Maxpool5 = nn.MaxPool1d(kernel_size = 2 , stride = 2) # 64

        self.DilatedConv6 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size = 3, padding = 'same', dilation = 2),
            nn.BatchNorm1d(512),
            nn.ReLU()
        ) 
        self.maxpool6 = nn.MaxPool1d(kernel_size = 2 , stride = 2) # 32


        # Attention Block
        self.MultiHeadSelfAttention1 = MultiHeadSelfAttention(dim = 512, num_heads = 8) # 헤드는 8개 차원은 512, 즉 각 해드당 차원은 64
        self.MultiHeadSelfAttention2 = MultiHeadSelfAttention(dim = 512, num_heads = 8) # 헤드는 8개 차원은 512, 즉 각 해드당 차원은 64

        # AdaptiveAvgPool1d
        self.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(output_size = 1) # 무조건 shape을 1로 만들어줌)



        # Classification Head

        self.classifier = nn.Sequential(
            nn.Linear(512 , 4),
            nn.Dropout(0.5),
        
        )

    
    def forward(self,  x):
        # [Batch , Seq_lenght , 1] = [64 , 2048, 1]
        x = x.flatten(1,2) # [64 , 2048]
        x = x.unsqueeze(1) # [64, 1, 2048]
        # B, N = x.shape #x.shape: torch.Size([128, 1024])
        # x = x.unsqueeze(1) # [128, 1, 1024]
        # Conv1 Block
        x_1 = self.DilatedConv1(x) # [64, 16, 2048]
        x_1 = self.Maxpool1(x_1) # [64, 16, 1024]

        # Conv2 Block
        x_2 = self.DilatedConv2(x_1) # [64, 32, 1024]
        x_2 = self.Maxpool2(x_2) # [64, 32, 512]

        # Conv3 Block
        x_3 = self.DilatedConv3(x_2) # [64, 64, 512]
        x_3 = self.Maxpool3(x_3) # [64, 64, 256]

        # Conv4 Block
        x_4 = self.DilatedConv4(x_3) # [64, 128, 256]
        x_4 = self.Maxpool4(x_4) # [64, 128, 128]

        # Conv5 Block
        x_5 = self.DilatedConv5(x_4) # [64, 256, 128]   
        x_5 = self.Maxpool5(x_5) # [64, 256, 64]

        # Conv6 Block
        x_6 = self.DilatedConv6(x_5) # [64, 512, 64]
        x_6 = self.maxpool6(x_6) # [64, 512, 32]

        # Attention block -> dim = 512, num_heads =8
        att1 = self.MultiHeadSelfAttention1(x_6) # [64, 512, 32]
        att2 = self.MultiHeadSelfAttention2(att1) # [64, 512, 32]

        # AdaptiveAvgPool1d
        x_att = self.AdaptiveAvgPool1d(att2) # [64, 512, 1] 

        # Classification Head
        x_att = x_att.flatten(1,2) # [64, 512]
        x_att = self.classifier(x_att) # [64, 10]

        return x_att
        
        
        
        
        
        
        
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim , num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.q_linear = nn.Sequential(
            nn.Conv1d(dim , dim , kernel_size = 1),
            nn.BatchNorm1d(dim)
            
        )        

        self.k_linear = nn.Sequential(
            nn.Conv1d(dim , dim , kernel_size = 1),
            nn.BatchNorm1d(dim)
            )
        
        self.v_linear = nn.Sequential(
            nn.Conv1d(dim , dim , kernel_size = 1),
            nn.BatchNorm1d(dim)
            
        )

        self.proj_conv = nn.Sequential(
            nn.Conv1d(dim , dim , kernel_size = 1),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )
    def forward(self, x):

        B, C, N = x.shape
        x_original = x

        q = self.q_linear(x)
        q = (
            q.transpose(1,2)
            .reshape(B , N , self.num_heads , C // self.num_heads)
            .permute(0, 1, 3, 2)
        ) 
        # -> [Batch , Head , Seq_len , Dims]

        k = self.k_linear(x)
        k = (
            k.transpose(1,2)
            .reshape(B , N , self.num_heads , C // self.num_heads)
            .permute(0, 1, 3, 2)
        )

        v = self.v_linear(x)
        v = (
            v.transpose(1,2)
            .reshape(B , N , self.num_heads , C // self.num_heads)
            .permute(0, 1, 3, 2)
        )

        attn_score = torch.matmul(q, k.transpose(-2 , -1))
        attn_score = attn_score / math.sqrt(C)
        attn_score = F.softmax(attn_score, dim = -1)

        out = torch.matmul(attn_score, v)
        out = out.permute(0, 1, 3, 2)
        out = out.reshape(B , C , N)

        out = self.proj_conv(out)
        out = out + x_original

        return out
        
        