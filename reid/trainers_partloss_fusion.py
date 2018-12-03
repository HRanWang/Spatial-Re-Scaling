from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
from .utils import Bar
from torch.nn import functional as F

class BaseTrainer(object):
    def __init__(self, model, criterion, X, Y, SMLoss_mode=0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader))
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9, loss10, loss11, loss12, loss13, loss14, loss15, loss16, loss17, loss18, loss19, loss20, loss21, loss22, loss23, prec1 = self._forward(inputs, targets)
            # loss0, loss1, loss2, loss3, loss4, loss5, prec1 = self._forward(inputs, targets)
#===================================================================================
            loss = (loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11+loss12+loss13+loss14+loss15+loss16+loss17+loss18+loss19+loss20+loss21+loss22+loss23)/24
            # loss = (loss0+loss1+loss2+loss3+loss4+loss5)/6
            # loss = 0
            # for i in range(24):
            #     loss = loss + losses[i]
            # loss = loss / 24
            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            # torch.autograd.backward([ loss0, loss1, loss2, loss3, loss4, loss5],
            #                         [torch.ones(1).cuda(), torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),
            #                          ])
            torch.autograd.backward([ loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9, loss10, loss11, loss12, loss13, loss14, loss15, loss16, loss17, loss18, loss19, loss20, loss21, loss22, loss23],
                                    [torch.ones(1).cuda(), torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),
                                     torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),
                                     torch.ones(1).cuda(), torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),
                                     torch.ones(1).cuda(), torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),
                                     ])
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bt:.3f} {N_bta:.3f} | Data {N_dt:.3f} {N_dta:.3f} | Loss {N_loss:.3f} {N_lossa:.3f} | Prec {N_prec:.2f} {N_preca:.2f}'.format(
                      N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),
                              N_bt=batch_time.val, N_bta=batch_time.avg,
                              N_dt=data_time.val, N_dta=data_time.avg,
                              N_loss=losses.val, N_lossa=losses.avg,
                              N_prec=precisions.val, N_preca=precisions.avg,
							  )
            bar.next()
        bar.finish()



    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        index = (targets-751).data.nonzero().squeeze_()

        losses = []
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            # for i in range(24):
            #     loss = self.criterion(outputs[i],targets)
            #     losses.append(loss)
            loss0 = self.criterion(outputs[1][0],targets)
            loss1 = self.criterion(outputs[1][1],targets)
            loss2 = self.criterion(outputs[1][2],targets)
            loss3 = self.criterion(outputs[1][3],targets)
            loss4 = self.criterion(outputs[1][4],targets)
            loss5 = self.criterion(outputs[1][5],targets)
            loss6 = self.criterion(outputs[1][6], targets)
            loss7 = self.criterion(outputs[1][7],targets)
            loss8 = self.criterion(outputs[1][8],targets)
            loss9 = self.criterion(outputs[1][9],targets)
            loss10 = self.criterion(outputs[1][10],targets)
            loss11 = self.criterion(outputs[1][11],targets)
            loss12 = self.criterion(outputs[1][12],targets)
            loss13 = self.criterion(outputs[1][13], targets)
            loss14 = self.criterion(outputs[1][14],targets)
            loss15 = self.criterion(outputs[1][15],targets)
            loss16 = self.criterion(outputs[1][16],targets)
            loss17 = self.criterion(outputs[1][17],targets)
            loss18 = self.criterion(outputs[1][18],targets)
            loss19 = self.criterion(outputs[1][19],targets)
            loss20 = self.criterion(outputs[1][20], targets)
            loss21 = self.criterion(outputs[1][21],targets)
            loss22 = self.criterion(outputs[1][22],targets)
            loss23 = self.criterion(outputs[1][23],targets)

            prec, = accuracy(outputs[1][2].data, targets.data)
            prec = prec[0]
                        
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        # return loss0,loss1,loss2,loss3,loss4,loss5, prec
        return loss0,loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,loss9,loss10,loss11,loss12,loss13,loss14,loss15,loss16,loss17,loss18,loss19,loss20,loss21,loss22,loss23, prec
