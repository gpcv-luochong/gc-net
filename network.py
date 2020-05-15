import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class BasicBlock(nn.Module):  #basic block for Conv2d
    def __init__(self,in_planes,planes,stride=1):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(planes)
        self.shortcut=nn.Sequential()
    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out=F.relu(out)
        return out


class ThreeDConv(nn.Module):
    def __init__(self,in_planes,planes,stride=1):
        super(ThreeDConv, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3=nn.Conv3d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm3d(planes)

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.relu(self.bn2(self.conv2(out)))
        out=F.relu(self.bn3(self.conv3(out)))
        return out


class GC_NET(nn.Module):
    def __init__(self,block,block_3d,num_block,height,width,maxdisp):
        super(GC_NET, self).__init__()
        self.height = height
        self.width = width
        self.maxdisp = int(maxdisp/2)
        self.in_planes = 32
        # first two conv2d
        self.conv0 = nn.Conv2d(3, 32, 5, 2, 2)
        self.bn0 = nn.BatchNorm2d(32)
        # res block
        self.res_block = self._make_layer(block, self.in_planes, 32, num_block[0], stride=1)
        # last conv2d
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)

        # conv3d
        self.conv3d_1 = nn.Conv3d(64, 32, 3, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(32)
        self.conv3d_2 = nn.Conv3d(32, 32, 3, 1, 1)
        self.bn3d_2 = nn.BatchNorm3d(32)

        self.conv3d_3 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_3 = nn.BatchNorm3d(64)
        self.conv3d_4 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_4 = nn.BatchNorm3d(64)
        self.conv3d_5 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_5 = nn.BatchNorm3d(64)

        # conv3d sub_sample block
        self.block_3d_1 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_2 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_3 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_4 = self._make_layer(block_3d, 64, 128, num_block[1], stride=2)

        # deconv3d
        self.deconv1 = nn.ConvTranspose3d(128, 64, 3, 2, 1, 1)
        self.debn1 = nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn2 = nn.BatchNorm3d(64)
        self.deconv3 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn3 = nn.BatchNorm3d(64)
        self.deconv4 = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.debn4 = nn.BatchNorm3d(32)

        # last deconv3d
        self.deconv5 = nn.ConvTranspose3d(32, 1, 3, 2, 1, 1)

        self.regression = DisparityRegression(maxdisp)

    def forward(self, imgLeft, imgRight):
        original_size = [1, self.maxdisp*2, imgLeft.size(2), imgLeft.size(3)]
        imgl0 = F.relu(self.bn0(self.conv0(imgLeft)))
        imgr0 = F.relu(self.bn0(self.conv0(imgRight)))

        imgl_block = self.res_block(imgl0)
        imgr_block = self.res_block(imgr0)

        imgl1 = self.conv1(imgl_block)
        imgr1 = self.conv1(imgr_block)
                # cost volume
        cost_volum = self.cost_volume(imgl1, imgr1)
            # print(cost_volum.shape)
        conv3d_out = F.relu(self.bn3d_1(self.conv3d_1(cost_volum)))
        conv3d_out = F.relu(self.bn3d_2(self.conv3d_2(conv3d_out)))
        # conv3d block
        conv3d_block_1 = self.block_3d_1(cost_volum)
        conv3d_21 = F.relu(self.bn3d_3(self.conv3d_3(cost_volum)))
        conv3d_block_2 = self.block_3d_2(conv3d_21)
        conv3d_24 = F.relu(self.bn3d_4(self.conv3d_4(conv3d_21)))
        conv3d_block_3 = self.block_3d_3(conv3d_24)
        conv3d_27 = F.relu(self.bn3d_5(self.conv3d_5(conv3d_24)))
        conv3d_block_4 = self.block_3d_4(conv3d_27)

        # deconv
        deconv3d = F.relu(self.debn1(self.deconv1(conv3d_block_4)) + conv3d_block_3)
        deconv3d = F.relu(self.debn2(self.deconv2(deconv3d)) + conv3d_block_2)
        deconv3d = F.relu(self.debn3(self.deconv3(deconv3d)) + conv3d_block_1)
        deconv3d = F.relu(self.debn4(self.deconv4(deconv3d)) + conv3d_out)

        # last deconv3d
        deconv3d = self.deconv5(deconv3d)
        out = deconv3d.view( original_size)
        prob = F.softmax(-out, 1)


        disp1 = self.regression(prob)

        return disp1



    def _make_layer(self,block,in_planes,planes,num_block,stride):
        strides=[stride]+[1]*(num_block-1)
        layers=[]
        for step in strides:
            layers.append(block(in_planes,planes,step))
        return nn.Sequential(*layers)

    def cost_volume(self,imgl,imgr):
        B, C, H, W = imgl.size()
        cost_vol = torch.zeros(B, C * 2, self.maxdisp , H, W).type_as(imgl)
        for i in range(self.maxdisp):
            if i > 0:
                cost_vol[:, :C, i, :, i:] = imgl[:, :, :, i:]
                cost_vol[:, C:, i, :, i:] = imgr[:, :, :, :-i]
            else:
                cost_vol[:, :C, i, :, :] = imgl
                cost_vol[:, C:, i, :, :] = imgr
        return cost_vol

    def __init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

def GcNet(height, width, maxdisp):
    return GC_NET(BasicBlock, ThreeDConv, [8, 1], height, width, maxdisp)
    # return GC_NET_new(BasicBlock, [8, 1], height, width, maxdisp)

class GC_NET_new(nn.Module):
    def __init__(self,block,num_block,height,width,maxdisp):
        super(GC_NET_new, self).__init__()
        self.height = height
        self.width = width
        self.maxdisp = int(maxdisp / 2)
        self.in_planes = 32
        # first two conv2d
        self.conv0 = nn.Conv2d(3, 32, 5, 2, 2)
        self.bn0 = nn.BatchNorm2d(32)
        # res block
        self.res_block = self._make_layer(block, self.in_planes, 32, num_block[0], stride=1)
        # last conv2d
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)

        #Cost Volume
        self.conv_gru1 = ConvGRUCell(32, 16, 3)
        self.conv_gru2 = ConvGRUCell(16, 4, 3)
        self.conv_gru3 = ConvGRUCell(4, 1, 3)
        self.prob_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)


    def forward(self, imgLeft, imgRight):
        imgl0 = F.relu(self.bn0(self.conv0(imgLeft)))
        imgr0 = F.relu(self.bn0(self.conv0(imgRight)))

        imgl_block = self.res_block(imgl0)
        imgr_block = self.res_block(imgr0)

        ref_tower = self.conv1(imgl_block)
        view_tower = self.conv1(imgr_block)

        feature_shape = ref_tower[0].shape
        batch_size = feature_shape[0]
        print(feature_shape)
        state1 = torch.zeros(batch_size, feature_shape[1], feature_shape[2], 16)
        state2 = torch.zeros(batch_size, feature_shape[1], feature_shape[2], 4)
        state3 = torch.zeros(batch_size, feature_shape[1], feature_shape[2], 2)


        ave_feature = ref_tower
        ave_feature2 = torch.pow(ref_tower,2)
        ave_feature = ave_feature + view_tower
        ave_feature2 = ave_feature2 + torch.pow(view_tower,2)
        cost = ave_feature2 - torch.pow(ave_feature,2)


        reg_cost1, state1 = self.conv_gru1(-cost, state1)
        reg_cost2, state2 = self.conv_gru2(reg_cost1, state2)
        reg_cost3, state3 = self.conv_gru3(reg_cost2, state3)
        disp_cost = self.prob_conv(reg_cost3)

        pro = nn.Softmax(disp_cost)
        disp1 = self.regression(pro)

        return disp1

    def _make_layer(self,block,in_planes,planes,num_block,stride):
        strides=[stride]+[1]*(num_block-1)
        layers=[]
        for step in strides:
            layers.append(block(in_planes,planes,step))
        return nn.Sequential(*layers)

class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvGRUCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)

        self.conv_gates = nn.Conv2d(self.input_channels + self.hidden_channels, 2 * self.hidden_channels,
                                    kernel_size=self.kernel_size, stride=1,
                                    padding=self.padding, bias=True)
        self.convc = nn.Conv2d(self.input_channels + self.hidden_channels, self.hidden_channels,
                               kernel_size=self.kernel_size, stride=1,
                               padding=self.padding, bias=True)

    def forward(self, x, h):
        input = torch.cat((x, h), dim=1)
        gates = self.conv_gates(input)

        reset_gate, update_gate = torch.chunk(gates, dim=1, chunks=2)

        # activation
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        # print(reset_gate)
        # concatenation
        input = torch.cat((x, reset_gate * h), dim=1)

        # convolution
        conv = self.convc(input)

        # activation
        conv = torch.tanh(conv)

        # soft update
        output = update_gate * h + (1 - update_gate) * conv

        return output, output

class DisparityRegression(nn.Module):

    def __init__(self, max_disp):
        super().__init__()

        self.disp_score = torch.range(0, max_disp - 1)  # [D]
        self.disp_score = self.disp_score.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, D, 1, 1]


    def forward(self, prob):
        disp_score = self.disp_score.expand_as(prob).type_as(prob)  # [B, D, H, W]
        out = torch.sum(disp_score * prob, dim=1)  # [B, H, W]
        return out
