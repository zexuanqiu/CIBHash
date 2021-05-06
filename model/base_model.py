import math
import torch
import random
import pickle
import sklearn
import argparse
import numpy as np
import seaborn as sb
from PIL import Image
import torch.nn as nn
from copy import deepcopy
from datetime import timedelta
from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
import matplotlib.patheffects as pe
from collections import OrderedDict
from torch.autograd import Variable
from sklearn.datasets import load_digits
from timeit import default_timer as timer


from utils.logger import Logger
from utils.data import LabeledData
from utils.evaluation import calculate_hamming
from utils.evaluation import compress, calculate_top_map

class Base_Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.load_data()
    
    def load_data(self):
        self.data = LabeledData(self.hparams.dataset)
    
    def get_hparams_grid(self):
        raise NotImplementedError

    def define_parameters(self):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def run_training_sessions(self):
        logger = Logger(self.hparams.model_path + '.log', on=True)
        val_perfs = []
        best_val_perf = float('-inf')
        start = timer()
        random.seed(self.hparams.seed)  # For reproducible random runs

        for run_num in range(1, self.hparams.num_runs + 1):
            state_dict, val_perf = self.run_training_session(run_num, logger)
            val_perfs.append(val_perf)

            if val_perf > best_val_perf:
                best_val_perf = val_perf
                logger.log('----New best {:8.4f}, saving'.format(val_perf))
                torch.save({'hparams': self.hparams,
                            'state_dict': state_dict}, self.hparams.model_path)
        
        logger.log('Time: %s' % str(timedelta(seconds=round(timer() - start))))
        self.load()
        if self.hparams.num_runs > 1:
            logger.log_perfs(val_perfs)
            logger.log('best hparams: ' + self.flag_hparams())
        
        val_perf, test_perf = self.run_test()
        logger.log('Val:  {:8.4f}'.format(val_perf))
        logger.log('Test: {:8.4f}'.format(test_perf))
    
    def run_training_session(self, run_num, logger):
        self.train()
        
        # Scramble hyperparameters if number of runs is greater than 1.
        if self.hparams.num_runs > 1:
            logger.log('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
            for hparam, values in self.get_hparams_grid().items():
                assert hasattr(self.hparams, hparam)
                self.hparams.__dict__[hparam] = random.choice(values)
        
        random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)

        self.define_parameters()

        # if encode_length is 16, then al least 80 epochs!
        if self.hparams.encode_length == 16:
            self.hparams.epochs = max(80, self.hparams.epochs)

        logger.log('hparams: %s' % self.flag_hparams())
        
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        self.to(device)

        optimizer = self.configure_optimizers()
        train_loader, val_loader, _, database_loader = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=True, get_test=False)
        best_val_perf = float('-inf')
        best_state_dict = None
        bad_epochs = 0

        try:
            for epoch in range(1, self.hparams.epochs + 1):
                forward_sum = {}
                num_steps = 0
                for batch_num, batch in enumerate(train_loader):
                    optimizer.zero_grad()

                    imgi, imgj, _ = batch
                    imgi = imgi.to(device)
                    imgj = imgj.to(device)

                    forward = self.forward(imgi, imgj, device)

                    for key in forward:
                        if key in forward_sum:
                            forward_sum[key] += forward[key]
                        else:
                            forward_sum[key] = forward[key]
                    num_steps += 1

                    if math.isnan(forward_sum['loss']):
                        logger.log('Stopping epoch because loss is NaN')
                        break

                    forward['loss'].backward()
                    optimizer.step()

                if math.isnan(forward_sum['loss']):
                    logger.log('Stopping training session because loss is NaN')
                    break
                
                logger.log('End of epoch {:3d}'.format(epoch), False)
                logger.log(' '.join([' | {:s} {:8.4f}'.format(
                    key, forward_sum[key] / num_steps)
                                     for key in forward_sum]), True)

                if epoch % self.hparams.validate_frequency == 0:
                    print('evaluating...')
                    val_perf = self.evaluate(database_loader, val_loader, self.data.topK, device)
                    logger.log(' | val perf {:8.4f}'.format(val_perf), False)

                    if val_perf > best_val_perf:
                        best_val_perf = val_perf
                        bad_epochs = 0
                        logger.log('\t\t*Best model so far, deep copying*')
                        best_state_dict = deepcopy(self.state_dict())
                    else:
                        bad_epochs += 1
                        logger.log('\t\tBad epoch %d' % bad_epochs)

                    if bad_epochs > self.hparams.num_bad_epochs:
                        break

        except KeyboardInterrupt:
            logger.log('-' * 89)
            logger.log('Exiting from training early')

        return best_state_dict, best_val_perf
    
    def evaluate(self, database_loader, val_loader, topK, device):
        self.eval()
        with torch.no_grad():
            retrievalB, retrievalL, queryB, queryL = compress(database_loader, val_loader, self.encode_discrete, device)
            result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=topK)
        self.train()
        return result
    
    def load(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        checkpoint = torch.load(self.hparams.model_path) if self.hparams.cuda \
                     else torch.load(self.hparams.model_path,
                                     map_location=torch.device('cpu'))
        if checkpoint['hparams'].cuda and not self.hparams.cuda:
            checkpoint['hparams'].cuda = False
        self.hparams = checkpoint['hparams']
        self.define_parameters()
        self.load_state_dict(checkpoint['state_dict'])
        self.to(device)
    
    def run_test(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        _, val_loader, test_loader, database_loader = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=False, get_test=True)
        
        val_perf = self.evaluate(database_loader, val_loader, self.data.topK, device)
        test_perf = self.evaluate(database_loader, test_loader, self.data.topK, device)
        return val_perf, test_perf
    
    def run_retrieval_case_study(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        query_idxs = [0,2,5]
        X_database = self.data.X_database
        X_test = self.data.X_test
        X_case = torch.cat([self.data.test_cifar10_transforms(Image.fromarray(self.data.X_test[i])).unsqueeze(0) for i in query_idxs], dim=0)
        _, val_loader, test_loader, database_loader = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=False, get_test=True)
        
        # get hash codes
        self.eval()
        with torch.no_grad():
            retrievalB = list([])
            for batch_step, (data, target) in enumerate(database_loader):
                var_data = Variable(data.to(device))
                code = self.encode_discrete(var_data)
                retrievalB.extend(code.cpu().data.numpy())

            queryB = list([])
            var_data = Variable(X_case.to(device))
            code = self.encode_discrete(var_data)
            queryB.extend(code.cpu().data.numpy())

        retrievalB = np.array(retrievalB)
        queryB = np.array(queryB)

        # get top 10 index
        top10_idx_list = []
        for idx in range(queryB.shape[0]):
            hamm = calculate_hamming(queryB[idx, :], retrievalB)
            ind = list(np.argsort(hamm)[:10])
            top10_idx_list.append(ind)

        # plot results
        fig = plt.figure(0, figsize = (5,1.2))
        fig.clf()
        gs = gridspec.GridSpec(queryB.shape[0], 12)
        gs.update(wspace = 0.0001, hspace = 0.0001)
        for i in range(queryB.shape[0]):
            axes = plt.subplot(gs[i,0])
            axes.imshow(X_test[query_idxs[i]])
            axes.axis('off')

            for j in range(0, 10):
                axes = plt.subplot(gs[i, j+2])
                axes.imshow(X_database[top10_idx_list[i][j]])
                axes.axis('off')
        fig.savefig("retrieval_case_study_{:d}bits.pdf".format(self.hparams.encode_length), bbox_inches='tight', pad_inches=0.0)

    def hash_code_visualization(self):
        """
        cifar10 labels:
        0: Airplane 1: Automobile 2: Bird 3: Cat 4: Deer
        5: Dog 6: Frog 7: Horse 8: Ship 9: Truck
        """
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        _, _, test_loader, _ = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=False, get_test=True)
        
        retrievalB = list([])
        retrievalL = list([])
        for batch_step, (data, target) in enumerate(test_loader):
            var_data = Variable(data.to(device))
            code = self.encode_discrete(var_data)
            retrievalB.extend(code.cpu().data.numpy())
            retrievalL.extend(target.cpu().data.numpy())

        hash_codes = np.array(retrievalB)
        _, labels = np.where(np.array(retrievalL) == 1)
        labels_ticks = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        
        # TSN
        mapper = TSNE(perplexity=30).fit_transform(hash_codes)

        plt.figure(figsize=(8, 8))
        plt.scatter(mapper[:,0], mapper[:,1], lw=0, s=20, c=labels.astype(np.int), cmap='Spectral')
        # cbar = plt.colorbar(boundaries=np.arange(11)-0.5, fraction=0.046, pad=0.04)
        # cbar.set_ticks(np.arange(10))
        # cbar.set_ticklabels(labels_ticks)

        # Add the labels for each digit.
        for i in range(10):
            # Position of each label.
            xtext, ytext = np.median(mapper[labels == i, :], axis=0)
            txt = plt.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        
        plt.axis("off")
        plt.gcf().tight_layout()
        plt.savefig('Ours_hash_codes_visulization_{:d}bits.pdf'.format(self.hparams.encode_length), bbox_inches='tight', pad_inches=0.0)

    def flag_hparams(self):
        flags = '%s' % (self.hparams.model_path)
        for hparam in vars(self.hparams):
            val = getattr(self.hparams, hparam)
            if str(val) == 'False':
                continue
            elif str(val) == 'True':
                flags += ' --%s' % (hparam)
            elif str(hparam) in {'model_path', 'num_runs',
                                 'num_workers'}:
                continue
            else:
                flags += ' --%s %s' % (hparam, val)
        return flags
    
    @staticmethod
    def get_general_hparams_grid():
        grid = OrderedDict({
            'seed': list(range(100000)),
            'lr': [0.003, 0.001, 0.0003, 0.0001],
            'batch_size': [64, 128, 256],
            })
        return grid

    @staticmethod
    def get_general_argparser():
        parser = argparse.ArgumentParser()

        parser.add_argument('model_path', type=str)
        parser.add_argument('--train', action='store_true',
                            help='train a model?')
        parser.add_argument('-d', '--dataset', default = 'cifar10', type=str,
                            help='dataset [%(default)s]')
        parser.add_argument("-l","--encode_length", type = int, default=16,
                            help = "Number of bits of the hash code [%(default)d]")
        parser.add_argument("--lr", default = 1e-3, type = float,
                            help='initial learning rate [%(default)g]')
        parser.add_argument("--batch_size", default=64,type=int,
                            help='batch size [%(default)d]')
        parser.add_argument("-e","--epochs", default=60, type=int,
                            help='max number of epochs [%(default)d]')
        parser.add_argument('--cuda', action='store_true',
                            help='use CUDA?')
        parser.add_argument('--num_runs', type=int, default=1,
                            help='num random runs (not random if 1) '
                            '[%(default)d]')
        parser.add_argument('--num_bad_epochs', type=int, default=6,
                            help='num indulged bad epochs [%(default)d]')
        parser.add_argument('--validate_frequency', type=int, default=20,
                            help='validate every [%(default)d] epochs')
        parser.add_argument('--num_workers', type=int, default=8,
                            help='num dataloader workers [%(default)d]')
        parser.add_argument('--seed', type=int, default=8888,
                            help='random seed [%(default)d]')
        parser.add_argument('--device', type=int, default=0, 
                            help='device of the gpu')
        
        
        return parser
