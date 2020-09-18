import torch
import torch.nn as nn
from torchsummary import summary

class CAE_pytorch(nn.Module):
    def __init__(self, in_channels = 1, rep_dim = 256):
        super(CAE_pytorch, self).__init__()
        # nf = 16
        # nf = (16, 64, 256, 1024)
        nf = [16, 32, 64, 128, 256]
        self.nf = nf

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf[0], kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(num_features=nf[0])
        self.enc_act1 = nn.ReLU(inplace=True)

        self.enc_conv2 = nn.Conv2d(in_channels=nf[0], out_channels=nf[1], kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(num_features=nf[1])
        self.enc_act2 = nn.ReLU(inplace=True)

        self.enc_conv3 = nn.Conv2d(in_channels=nf[1], out_channels=nf[2], kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(num_features=nf[2])
        self.enc_act3 = nn.ReLU(inplace=True)

        self.enc_conv4 = nn.Conv2d(in_channels=nf[2], out_channels=nf[3], kernel_size=3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(num_features=nf[3])
        self.enc_act4 = nn.ReLU(inplace=True)

        self.enc_conv5 = nn.Conv2d(in_channels=nf[3], out_channels=nf[4], kernel_size=3, stride=2, padding=1)
        self.enc_bn5 = nn.BatchNorm2d(num_features=nf[4])
        self.enc_act5 = nn.ReLU(inplace=True)

        # self.enc_fc = nn.Linear(nf * 4 * 16 * 16, rep_dim)
        # self.rep_act = nn.Tanh()

        # Decoder
        # self.dec_fc = nn.Linear(rep_dim, nf * 4 * 16 * 16)
        # self.dec_bn0 = nn.BatchNorm1d(num_features=nf * 4 * 16 * 16)
        # self.dec_act0 = nn.ReLU(inplace=True)
        
        self.dec_conv00 = nn.ConvTranspose2d(in_channels=nf[4], out_channels=nf[3], kernel_size=4, stride=2, padding=1, output_padding=0)
        self.dec_bn00 = nn.BatchNorm2d(num_features=nf[3])
        self.dec_act00 = nn.ReLU(inplace=True)

        self.dec_conv0 = nn.ConvTranspose2d(in_channels=nf[3], out_channels=nf[2], kernel_size=4, stride=2, padding=1, output_padding=0)
        self.dec_bn0 = nn.BatchNorm2d(num_features=nf[2])
        self.dec_act0 = nn.ReLU(inplace=True)

        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf[2], out_channels=nf[1], kernel_size=4, stride=2, padding=1, output_padding=0)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf[1])
        self.dec_act1 = nn.ReLU(inplace=True)

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf[1], out_channels=nf[0], kernel_size=4, stride=2, padding=1, output_padding=0)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf[0])
        self.dec_act2 = nn.ReLU(inplace=True)

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf[0], out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.output_act = nn.Sigmoid()

    def encode(self, x):
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_act4(self.enc_bn4(self.enc_conv4(x)))
        x = self.enc_act5(self.enc_bn5(self.enc_conv5(x)))
        # rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return x

    def decode(self, rep):
        # x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
        # x = x.view(-1, self.nf * 4, 16, 16)
        x = self.dec_act00(self.dec_bn00(self.dec_conv00(rep)))
        x = self.dec_act0(self.dec_bn0(self.dec_conv0(x)))
        x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
        x = self.output_act(self.dec_conv3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))
        # return torch.sigmoid(x)

if __name__ == "__main__":

    model = CAE_pytorch()
    print(model)

    x = torch.rand(1, 256, 256)
    summary(model.cuda(), (1, 256, 256))