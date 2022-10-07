import torch
import torch.nn as nn
from models.CoordASPP_atten import SPCS, SPCS_NoSPP


class PCAWSD(nn.Module):
    def __init__(self, num_classes, n_bands, chanel):
        super(PCAWSD, self).__init__()

        self.bands = n_bands
        chanel = chanel
        kernel = 5
        CCChannel = 25

        self.b1 = nn.BatchNorm2d(self.bands)
        self.con11 = nn.Conv2d(self.bands, chanel, 1, padding=0, bias=True)
        self.s1 = nn.Sigmoid()
        self.con12 = nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)

        self.nlcon2 = SPCS(chanel+self.bands, chanel+self.bands)
        self.b2 = nn.BatchNorm2d(self.bands + chanel)
        self.con21 = nn.Conv2d(self.bands + chanel, chanel, 1, padding=0, bias=True)
        self.s2 = nn.Sigmoid()
        self.con22 = nn.Conv2d(chanel, CCChannel, kernel, padding=2, groups=CCChannel, bias=True)
        self.resx2 = nn.Conv2d(self.bands, CCChannel, 1, bias=True)
        self.resnl2 = nn.Conv2d(self.bands, chanel + self.bands, 1, bias=True)

        self.nlcon4 = SPCS(CCChannel + self.bands, CCChannel + self.bands)
        self.b4 = nn.BatchNorm2d(CCChannel + self.bands)
        self.con41 = nn.Conv2d(CCChannel + self.bands, chanel, 1, padding=0, bias=True)
        self.s4 = nn.Sigmoid()
        self.con42 = nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.resx4 = nn.Conv2d(self.bands,chanel , 1, bias=True)
        self.resnl4 = nn.Conv2d(self.bands,CCChannel + self.bands , 1, bias=True)

        self.nlcon5 = SPCS(CCChannel + chanel, CCChannel + chanel)
        self.b5 = nn.BatchNorm2d(CCChannel + chanel)
        self.con5 = nn.Conv2d(CCChannel + chanel, chanel, 1, padding=0, bias=True)
        self.s5 = nn.Sigmoid()
        self.cond5 = nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd5 = nn.Sigmoid()

        self.con6 = nn.Conv2d(chanel + CCChannel, num_classes + 1, 1, padding=0, bias=True)

    def forward(self, x):
        n = x.size(0)
        H = x.size(2)
        W = x.size(3)

        out1 = self.b1(x)
        out1 = self.con11(out1)
        out1 = self.s1(out1)
        out1 = self.con12(out1)
        out1 = self.s1(out1)

        out2 = torch.cat((out1, x), 1)
        nl2 = self.nlcon2(out2)
        out2 = self.resnl2(x) + nl2
        out2 = self.b2(out2)
        out2 = self.con21(out2)
        out2 = self.s2(out2)
        out2 = self.con22(out2)
        out2 = self.s2(out2)

        out4 = torch.cat((out2, x), 1)
        nl4 = self.nlcon4(out4)
        out4 = self.resnl4(x) + nl4
        out4 = self.b4(out4)
        out4 = self.con41(out4)
        out4 = self.s4(out4)
        out4 = self.con42(out4)
        out4 = self.s4(out4)
        out4 = out4+self.resx4(x)


        out5 = torch.cat((out4, out2), 1)
        nl5 = self.nlcon5(out5)
        out5 = self.b5(nl5)
        out5 = self.con5(out5)
        out5 = self.s5(out5)
        out5 = self.cond5(out5)
        out5 = self.sd5(out5)

        out6 = torch.cat((out5, out2), 1)
        out6 = self.con6(out6)

        return out6


class PCAWSD_NoSPP(nn.Module):
    def __init__(self, num_classes, n_bands, chanel):
        super(PCAWSD_NoSPP, self).__init__()
        self.bands = n_bands
        chanel = chanel
        kernel = 5
        CCChannel = 25

        self.b1 = nn.BatchNorm2d(self.bands)
        self.con11 = nn.Conv2d(self.bands, chanel, 1, padding=0, bias=True)
        self.s1 = nn.Sigmoid()
        self.con12 = nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)

        self.nlcon2 = SPCS_NoSPP(chanel + self.bands, chanel + self.bands)
        self.b2 = nn.BatchNorm2d(self.bands + chanel)
        self.con21 = nn.Conv2d(self.bands + chanel, chanel, 1, padding=0, bias=True)
        self.s2 = nn.Sigmoid()
        self.con22 = nn.Conv2d(chanel, CCChannel, kernel, padding=2, groups=CCChannel, bias=True)
        self.resx2 = nn.Conv2d(self.bands, CCChannel, 1, bias=True)
        self.resnl2 = nn.Conv2d(self.bands, chanel + self.bands, 1, bias=True)

        self.nlcon4 = SPCS_NoSPP(CCChannel + self.bands, CCChannel + self.bands)
        self.b4 = nn.BatchNorm2d(CCChannel + self.bands)
        self.con41 = nn.Conv2d(CCChannel + self.bands, chanel, 1, padding=0, bias=True)
        self.s4 = nn.Sigmoid()
        self.con42 = nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.resx4 = nn.Conv2d(self.bands, chanel, 1, bias=True)
        self.resnl4 = nn.Conv2d(self.bands, CCChannel + self.bands, 1, bias=True)

        self.nlcon5 = SPCS_NoSPP(CCChannel + chanel, CCChannel + chanel)
        self.b5 = nn.BatchNorm2d(CCChannel + chanel)
        self.con5 = nn.Conv2d(CCChannel + chanel, chanel, 1, padding=0, bias=True)
        self.s5 = nn.Sigmoid()
        self.cond5 = nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd5 = nn.Sigmoid()

        self.con6 = nn.Conv2d(chanel + CCChannel, num_classes + 1, 1, padding=0, bias=True)

    def forward(self, x):
        n = x.size(0)
        H = x.size(2)
        W = x.size(3)

        out1 = self.b1(x)
        out1 = self.con11(out1)
        out1 = self.s1(out1)
        out1 = self.con12(out1)
        out1 = self.s1(out1)

        out2 = torch.cat((out1, x), 1)
        nl2 = self.nlcon2(out2)
        out2 = self.resnl2(x) + nl2
        out2 = self.b2(out2)
        out2 = self.con21(out2)
        out2 = self.s2(out2)
        out2 = self.con22(out2)
        out2 = self.s2(out2)

        out4 = torch.cat((out2, x), 1)
        nl4 = self.nlcon4(out4)
        out4 = self.resnl4(x) + nl4
        out4 = self.b4(out4)
        out4 = self.con41(out4)
        out4 = self.s4(out4)
        out4 = self.con42(out4)
        out4 = self.s4(out4)
        out4 = out4 + self.resx4(x)

        out5 = torch.cat((out4, out2), 1)
        nl5 = self.nlcon5(out5)
        out5 = self.b5(nl5)
        out5 = self.con5(out5)
        out5 = self.s5(out5)
        out5 = self.cond5(out5)
        out5 = self.sd5(out5)

        out6 = torch.cat((out5, out2), 1)
        out6 = self.con6(out6)

        return out6
