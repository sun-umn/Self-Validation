import torch.nn as nn


############################ Encoder Net ##################################
class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.encoder_channels = [3, 32, 64, 128, 128, 128, 128, 1]
        self.code_dim = 16
        ################################## define EncoderNet #####################
        self.encoderNet = nn.Sequential(
            ##### 512->256
            nn.Conv2d(self.encoder_channels[0], self.encoder_channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.encoder_channels[1]),
            nn.ReLU(inplace=True),

            ##### 256->128
            nn.Conv2d(self.encoder_channels[1], self.encoder_channels[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.encoder_channels[2]),
            nn.ReLU(inplace=True),

            ##### 128->64
            nn.Conv2d(self.encoder_channels[2], self.encoder_channels[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.encoder_channels[3]),
            nn.ReLU(inplace=True),

            ##### 64->32
            nn.Conv2d(self.encoder_channels[3], self.encoder_channels[4], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.encoder_channels[4]),
            nn.ReLU(inplace=True),

            ##### 32->16
            nn.Conv2d(self.encoder_channels[4], self.encoder_channels[5], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.encoder_channels[5]),
            nn.ReLU(inplace=True),

            ##### 16->8
            nn.Conv2d(self.encoder_channels[5], self.encoder_channels[6], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.encoder_channels[6]),
            nn.ReLU(inplace=True),

            ##### 8->4
            nn.Conv2d(self.encoder_channels[6], self.encoder_channels[7], kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(self.encoder_channels[7]),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_data):
        ############### processed by encoder
        out_encoder = self.encoderNet(input_data)
        out_encoder = out_encoder.view(-1, self.code_dim)
        return out_encoder


###################### Define Deep Linear Net ##################
class DeepL(nn.Module):
    def __init__(self):
        super(DeepL, self).__init__()
        self.code_dim = 16

        ############################# Define Deep Linear Layers #######################
        self.linearNet = nn.Sequential(
            nn.Linear(self.code_dim, self.code_dim, bias=False),
            nn.Linear(self.code_dim, self.code_dim, bias=False),
            nn.Linear(self.code_dim, self.code_dim, bias=False),
            nn.Linear(self.code_dim, self.code_dim, bias=False),
        )

    def forward(self, input_data):
        out_linear = self.linearNet(input_data)
        return out_linear


###################### Define Decoder ##################
class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        self.decoder_channels = [1, 128, 128, 128, 128, 64, 32, 3]
        self.code_dim = 16
        ################################## define DecoderNet #####################
        self.decoderNet = nn.Sequential(
            ##### 4->8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.decoder_channels[0], self.decoder_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.decoder_channels[1]),
            nn.ReLU(inplace=True),

            ##### 8->16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.decoder_channels[1], self.decoder_channels[2], kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(self.decoder_channels[2]),
            nn.ReLU(inplace=True),

            ##### 16->32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.decoder_channels[2], self.decoder_channels[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.decoder_channels[3]),
            nn.ReLU(inplace=True),

            ##### 32->64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.decoder_channels[3], self.decoder_channels[4], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.decoder_channels[4]),
            nn.ReLU(inplace=True),

            ##### 64->128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.decoder_channels[4], self.decoder_channels[5], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.decoder_channels[5]),
            nn.ReLU(inplace=True),

            ##### 128->256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.decoder_channels[5], self.decoder_channels[6], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.decoder_channels[6]),
            nn.ReLU(inplace=True),

            ##### 256->512
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.decoder_channels[6], self.decoder_channels[7], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.decoder_channels[7]),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        ############### processed by decoder
        in_decoder = input_data.view(-1, 1, 4, 4)
        out_decoder = self.decoderNet(in_decoder)
        return out_decoder