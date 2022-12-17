"""
Graph Neural Network class for batched learning on image datasets
"""
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from function_transformer_attention import ODEFuncTransformerAtt
from function_GAT_attention import ODEFuncAtt
# from function_dorsey_attention import ODEFuncDorseyAtt
from function_laplacian_diffusion import LaplacianODEFunc
# from sde import SDEFunc, SDEblock
from block_transformer_attention import AttODEblock
from block_constant import ConstantODEblock
from block_mixed import MixedODEblock
from base_classes import BaseGNN
from model_configurations import set_block, set_function


class MNISTConvNet(nn.Module):
    def __init__(self, opt):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = opt['im_chan'], out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 16, 10) #120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 16)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

class CIFARConvNet(nn.Module):
    def __init__(self, opt):
        super(CIFARConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = opt['im_chan'], out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(5 * 5 * 16, 10) #20)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 5 * 5 * 16)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

# Define the GNN model.
class GNN_image_pixel(BaseGNN):
    def __init__(self, opt, data, num_classes, device=torch.device('cpu')):
        DataWrapper = namedtuple('DataWrapper', ['num_features'])
        dw = DataWrapper(1)
        DatasetWrapper = namedtuple('DatasetWrapper', ['data', 'num_classes'])
        dsw = DatasetWrapper(dw, num_classes)
        super(GNN_image_pixel, self).__init__(opt, dsw, device)
        self.f = set_function(opt)
        self.block = set_block(opt)
        self.data = data
        time_tensor = torch.tensor([0, self.T]).to(device)
        # self.odeblock = self.block(self.f, self.regularization_fns, opt, num_nodes, edge_index, edge_attr, device, t=time_tensor).to(self.device)

        self.odeblock = self.block(self.f, self.regularization_fns, opt, self.data, device, t=time_tensor).to(self.device)

        self.bn = nn.BatchNorm1d(num_features=opt['im_chan']) #hidden_dim']) #opt['im_chan']

        if self.opt['pixel_loss'] == '10catM2':
            self.m2 = nn.Linear(opt['im_chan'], opt['pixel_cat'])
        elif self.opt['pixel_loss'] == 'binary_sigmoid':
            self.m2 = None
            # self.m2 = nn.Linear(opt['hidden_dim'], opt['im_chan'])
        else:
            self.m2 = None
            # self.m2 = nn.Linear(opt['im_width'] * opt['im_height'] * opt['im_chan'], opt['pixel_cat'])
        # self.ConvNet = MNISTConvNet(opt) if opt['im_dataset'] == 'MNIST' else CIFARConvNet(opt)


    def forward(self, x):

        # x = self.m1(x)
        x = self.bn(x)
        self.odeblock.set_x0(x)


        z = self.odeblock(x)

        # Activation.
        # z = F.relu(z)

        # Dropout.
        # z = F.dropout(z, self.opt['dropout'], training=self.training)
        # if self.eval:  # keep batch norm as keps in -1,1 which is correct for sigmoid
        #   # z = z  * (self.bn.running_var + self.bn.eps)**0.5 + self.bn.running_mean
        #   z = (z - self.bn.bias) * (self.bn.running_var + self.bn.eps) ** 0.5 / self.bn.weight + self.bn.running_mean


        if self.opt['pixel_loss'] == 'binary_sigmoid':
            # if self.training: removed the drop outs as messing with scaling and sigmoid activation
            #   inputDropOutFactor = 1/(1-self.opt['input_dropout'])
            #   dropOutFactor = 1/(1-self.opt['dropout'])
            #   z = (z - self.bn.bias) * (self.bn.running_var + self.bn.eps) ** 0.5 / self.bn.weight + self.bn.running_mean
            #   z = (z - 0.5 * dropOutFactor) / (0.5 * dropOutFactor)
            # else:
            #   z = (z - 0.5) / 0.5 #MNIST in [0,1] and sigmoid(0)= 0.5 so need to rescale tp [-1,1]

            # z = self.m2(z)
            z = torch.cat((torch.sigmoid(z), 1 - torch.sigmoid(z)),dim=1)
        elif self.opt['pixel_loss'] == '10catM2':
            # z = z.view(-1, self.opt['im_chan'])
            z = self.m2(z) #decoder to number of classes this is like 10, 3->1 linear layers
            z = torch.sigmoid(z)
        elif self.opt['pixel_loss'] == '10catlogits'and self.opt['im_chan'] == 1:
            z = z.view(-1,1)
            cats = torch.arange(self.opt['pixel_cat']).to(self.device)
            z = 1 / ((z - cats) ** 2 + 1e-5)
            torch.cat((torch.sigmoid(z), 1 - torch.sigmoid(z)), dim=1)
        elif self.opt['pixel_loss'] == '10catkmeans':
            pass

        # elif self.opt['pixel_loss'] == 'MSE':
        #   z = z

        # for node (or pixel) classification
        # z = z.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])
        # z = self.m2(z) #decoder to number of classes

        #for graph classification
        # z = z.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])
        # z = self.m2(z)
        # for graph classification with conv
        # z = z.view(-1, self.opt['im_width'], self.opt['im_height'], self.opt['im_chan'])
        #to check image resizes
        # with torch.no_grad():
        #   import matplotlib.pyplot as plt
        #   fig = plt.figure()
        #   plt.imshow(z[0,:,:,:])
        #   plt.show()
        # z = torch.movedim(z,3,1)
        # z = z.permute(0,3,1,2)
        # z = self.ConvNet(z)

        return z

    def forward_plot_T(self, x): #the same as forward but without the decoder
        # Encode each node based on its feature.
        # x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        # x = self.m1(x) #no encoding for image viz
        # if True:
        #   x = F.relu(x)

        # Solve the initial value problem of the ODE.
        # if self.opt['augment']: #no augmenting for image viz
        #   c_aux = torch.zeros(x.shape).to(self.device)
        #   x = torch.cat([x, c_aux], dim=1)
        x = self.bn(x)
        self.odeblock.set_x0(x)

        if self.training:
            z, self.reg_states = self.odeblock(x)
        else:
            z = self.odeblock(x)

        # if self.opt['augment']: #no augmenting for image viz
        #   z = torch.split(z, x.shape[1] // 2, dim=1)[0]

        # Activation.
        # z = F.relu(z)

        # Dropout. not needed in eval
        # z = F.dropout(z, self.opt['dropout'], training=self.training)
        # z = z.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])

        #UNDO batch norm - see https: // pytorch.org / docs / stable / generated / torch.nn.BatchNorm1d.html
        if self.eval: #undo batch norm
            # z = z  * (self.bn.running_var + self.bn.eps)**0.5 + self.bn.running_mean
            z = (z - self.bn.bias) * (self.bn.running_var + self.bn.eps) ** 0.5 / self.bn.weight + self.bn.running_mean
        return z #output as 2d

    def forward_plot_path(self, x, frames): #stitch together ODE integrations
        # Encode each node based on its feature.
        # x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        # x = self.m1(x) #no encoding for image viz
        # if True:
        #   x = F.relu(x)

        # Solve the initial value problem of the ODE.
        # if self.opt['augment']: #no augmenting for image viz
        #   c_aux = torch.zeros(x.shape).to(self.device)
        #   x = torch.cat([x, c_aux], dim=1)

        paths = [x.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])]

        x = self.bn(x)
        if self.opt['block'] == 'attention':
            self.odeblock.x0_superpix = x

        z = x

        for f in range(frames):
            self.odeblock.set_x0(z) #(x)

            if self.opt['block'] == 'attention':
                z = self.odeblock.visualise(z)    #forward pass that does not recalc attentions on diffused X
            else:
                z = self.odeblock(z)

            # Activation.
            # z = F.relu(z)
            # Dropout.
            # z = F.dropout(z, self.opt['dropout'], training=self.training)

            if self.eval: #undo batch norm
                # path = z  * (self.bn.running_var + self.bn.eps)**0.5 + self.bn.running_mean
                path = (z-self.bn.bias) * (self.bn.running_var + self.bn.eps) ** 0.5 / self.bn.weight + self.bn.running_mean
                path = path.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])
                paths.append(path)

        paths = torch.stack(paths,dim=1)
        return paths #output as 1d


    def forward_plot_SuperPix(self, x, frames): #stitch together ODE integrations
        # x = self.m1(x)
        # atts = [self.odeblock.odefunc.attention_weights]  # edge_weight]
        if self.opt['function'] == 'transformer':
            atts = [self.odeblock.odefunc.edge_weight] #not calculated until 1st forward pass attention_weights]  # edge_weight]
        elif self.opt['block'] == 'attention':
            atts = [self.odeblock.odefunc.edge_weight]#attention_weights]
        else:
            atts = [self.odeblock.odefunc.edge_weight]
        paths = [x] #[torch.sigmoid(self.m2(x))] #[x]#.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])]

        x = self.bn(x)
        if self.opt['block'] == 'attention':
            self.odeblock.x0_superpix = x

        z = x
        for f in range(frames):
            # todo this is problematic for lin-att as each self.odeblock(z) forward pass
            #  does it recalc the attentions and therefore non-stationary/be subject to entropy?
            self.odeblock.set_x0(z) #(x)

            if self.opt['block'] == 'attention':
                z = self.odeblock.visualise(z)    #forward pass that does not recalc attentions on diffused X
            else:
                z = self.odeblock(z)

            if self.eval: #undo batch norm
                # path = path.view(-1, self.opt['im_width'] * self.opt['im_height'] * self.opt['im_chan'])
                path = (z-self.bn.bias) * (self.bn.running_var + self.bn.eps) ** 0.5 / self.bn.weight + self.bn.running_mean
                # path = torch.sigmoid(self.m2(z))
                paths.append(path)
                # atts.append(self.odeblock.odefunc.attention_weights) #edge_weight)
                if self.opt['function'] == 'transformer':
                    atts.append(self.odeblock.odefunc.attention_weights)
                elif self.opt['block'] == 'attention':
                    atts.append(self.odeblock.odefunc.attention_weights)#attention_weights)
                else:
                    atts.append(self.odeblock.odefunc.edge_weight)

        paths = torch.stack(paths,dim=1)
        # atts = torch.stack(atts,dim=1)

        return paths, atts