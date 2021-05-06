import math
import os
import statistics as stat
import sys

class Logger(object):

    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = False

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += '+'
    
    def log(self, string, newline=True):
        if self.on:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

        sys.stdout.write(string)
        if newline: sys.stdout.write('\n')
        sys.stdout.flush()
    
    def log_perfs(self, perfs):
        valid_perfs = [perf for perf in perfs if not math.isinf(perf)]
        best_perf = max(valid_perfs)
        self.log('-' * 89)
        self.log('%d perfs: %s' % (len(perfs), str(perfs)))
        self.log('perf max: %g' % best_perf)
        self.log('perf min: %g' % min(valid_perfs))
        self.log('perf avg: %g' % stat.mean(valid_perfs))
        self.log('perf std: %g' % (stat.stdev(valid_perfs)
                                     if len(valid_perfs) > 1 else 0.0))
        self.log('(excluded %d out of %d runs that produced -inf)' %
                 (len(perfs) - len(valid_perfs), len(perfs)))
        self.log('-' * 89)