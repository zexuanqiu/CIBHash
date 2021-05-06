import torch
import argparse
import torchvision
import torch.nn as nn
from torch.autograd import Function

from model.base_model import Base_Model

class CIBHash(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def define_parameters(self):
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        for param in self.vgg.parameters():
            param.requires_grad = False        
        self.encoder = nn.Sequential(nn.Linear(4096, 1024),
                                       nn.ReLU(),
                                       nn.Linear(1024, self.hparams.encode_length),
                                      )
        
        self.criterion = NtXentLoss(self.hparams.batch_size, self.hparams.temperature)
    
    def forward(self, imgi, imgj, device):
        imgi = self.vgg.features(imgi)
        imgi = imgi.view(imgi.size(0), -1)
        imgi = self.vgg.classifier(imgi)
        prob_i = torch.sigmoid(self.encoder(imgi))
        z_i = hash_layer(prob_i - 0.5)

        imgj = self.vgg.features(imgj)
        imgj = imgj.view(imgj.size(0), -1)
        imgj = self.vgg.classifier(imgj)
        prob_j = torch.sigmoid(self.encoder(imgj))
        z_j = hash_layer(prob_j - 0.5)

        kl_loss = (self.compute_kl(prob_i, prob_j) + self.compute_kl(prob_j, prob_i)) / 2
        contra_loss = self.criterion(z_i, z_j, device)
        loss = contra_loss + self.hparams.weight * kl_loss

        return {'loss': loss, 'contra_loss': contra_loss, 'kl_loss': kl_loss}
    
    def encode_discrete(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)

        prob = torch.sigmoid(self.encoder(x))
        z = hash_layer(prob - 0.5)

        return z

    def compute_kl(self, prob, prob_v):
        prob_v = prob_v.detach()
        # prob = prob.detach()

        kl = prob * (torch.log(prob + 1e-8) - torch.log(prob_v + 1e-8)) + (1 - prob) * (torch.log(1 - prob + 1e-8 ) - torch.log(1 - prob_v + 1e-8))
        kl = torch.mean(torch.sum(kl, axis = 1))
        return kl

    def configure_optimizers(self):
        return torch.optim.Adam([{'params': self.encoder.parameters()}], lr = self.hparams.lr)
    
    def get_hparams_grid(self):
        grid = Base_Model.get_general_hparams_grid()
        grid.update({
            'temperature': [0.2, 0.3, 0.4],
            'weight': [0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument("-t", "--temperature", default = 0.3, type = float,
                            help = "Temperature [%(default)d]",)
        parser.add_argument('-w',"--weight", default = 0.001, type=float,
                            help='weight of I(x,z) [%(default)f]')
        return parser


class hash(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data

        return grad_output

def hash_layer(input):
    return hash.apply(input)

class NtXentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NtXentLoss, self).__init__()
        #self.batch_size = batch_size
        self.temperature = temperature
        #self.device = device

        #self.mask = self.mask_correlated_samples(batch_size)
        self.similarityF = nn.CosineSimilarity(dim = 2)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
    

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    

    def forward(self, z_i, z_j, device):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        #sim = 0.5 * (z_i.shape[1] - torch.tensordot(z.unsqueeze(1), z.T.unsqueeze(0), dims = 2)) / z_i.shape[1] / self.temperature

        sim_i_j = torch.diag(sim, batch_size )
        sim_j_i = torch.diag(sim, -batch_size )
        
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)

        labels = torch.zeros(N).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss