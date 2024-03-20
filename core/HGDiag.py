import os
import time

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import shutil

from core.ita import cal_task_affinity
from core.loss.AutomaticWeightedLoss import AutomaticWeightedLoss
from core.model.MainModel import MainModel
from core.aug import *
from helper.eval import *
from helper.hg_util import batch_hypergraph
from .loss.DynamicNodeLoss import dynamicNodeLoss

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class HGDiag(object):

    def __init__(self, args, logger, device):
        self.args = args
        self.device = device
        self.logger = logger

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_period = args.eval_period
        self.log_every_n_steps = args.log_step
        self.tau = args.temperature
        log_dir = f"logs/{args.dataset}"
        os.makedirs(log_dir, exist_ok=True)

        # alias tensorboard='python3 -m tensorboard.main'
        # tensorboard --logdir=logs --host=192.168.31.201
        self.writer = SummaryWriter(log_dir)
        self.printParams()

    def printParams(self):
        self.logger.info(f"Training with: {self.device}")
        self.logger.info(f"Status of dynamic_weight: {self.args.dynamic_weight}")
        self.logger.info(f"lr: {self.args.lr}, weight_decay: {self.args.weight_decay}")

    def train(self, train_dl, test_data):

        model = MainModel(self.args).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        awl = AutomaticWeightedLoss(2)

        self.logger.info(model)
        self.logger.info(f"Start training for {self.epochs} epochs.")

        n_test = 0
        best_avg, best_f1 = 0, 0
        best_data = {}

        # Overhead
        fit_times = []
        test_times = []

        # Inter-Task Affinity (RCL -> FTI, FTI -> RCL)
        Z_r2fs, Z_f2rs = [], []

        for epoch in range(self.epochs):
            n_iter = 0
            start_time = time.time()
            model.train()
            epoch_loss = 0

            for batch_graphs, batch_hypergraphs, batch_labels in train_dl:
                instance_labels = batch_labels[:, 0]
                type_labels = batch_labels[:, 1]

                opt.zero_grad()

                root_logit, type_logit = model(batch_graphs, batch_hypergraphs)


                l_rcl = dynamicNodeLoss(batch_graphs,root_logit,instance_labels)
                l_fti = F.cross_entropy(type_logit, type_labels)
                if self.args.dynamic_weight:
                    total_loss = awl(l_rcl, l_fti)
                else:
                    total_loss = l_rcl + l_fti

                self.logger.debug("RCA_loss: {:.3f}, TC_loss: {:.3f}"
                                  .format(l_rcl, l_fti))

                # Calculate Inter-Task Affinity
                if epoch == 0:
                    Z_r2f, Z_f2r = cal_task_affinity(model=model,
                                                     optimizer=opt,
                                                     batch_graphs=batch_graphs,
                                                     batch_hypergraphs=batch_hypergraphs,
                                                     instance_labels=instance_labels,
                                                     type_labels=type_labels)
                    Z_r2fs.append(Z_r2f)
                    Z_f2rs.append(Z_f2r)

                total_loss.backward()
                # model.locator.clip_grad_norm()
                # grad = model.locator.fc_out.weight.grad.clone().detach().view(-1)
                # print(grad.cpu().numpy())
                opt.step()
                epoch_loss += total_loss.detach().item()
                n_iter += 1

            mean_epoch_loss = epoch_loss / n_iter
            end_time = time.time()
            time_per_epoch = (end_time - start_time)
            fit_times.append(time_per_epoch)
            self.logger.info("Epoch {} done. Loss: {:.3f}, Time per epoch: {:.3f}[s]"
                             .format(epoch, mean_epoch_loss, time_per_epoch))

            # .view(-1, 10)
            top1, top3, top5 = accuracy(root_logit.detach().cpu(), instance_labels.detach().cpu(), batch_graphs, topk=(1, 3, 5))
            type_logit = type_logit.detach().cpu()
            type_labels = type_labels.detach().cpu()
            pre = precision(type_logit, type_labels, k=5)
            rec = recall(type_logit, type_labels, k=5)
            f1 = f1score(type_logit, type_labels, k=5)
            self.writer.add_scalar('loss', epoch_loss / n_iter, global_step=epoch)
            self.writer.add_scalar('train/top1', top1, global_step=epoch)
            self.writer.add_scalar('train/top3', top3, global_step=epoch)
            self.writer.add_scalar('train/top5', top5, global_step=epoch)
            self.writer.add_scalar('train/precision', pre, global_step=epoch)
            self.writer.add_scalar('train/recall', rec, global_step=epoch)
            self.writer.add_scalar('train/f1-score', f1, global_step=epoch)

            # evaluate
            if epoch % self.eval_period == 0:
                n_test += 1
                graphs, hypergraphs, instance_labels, type_labels = [], [], [], []

                for data in test_data:
                    graphs.append(data[0])
                    hypergraphs.append(data[1])
                    instance_labels.append(data[2][0])
                    type_labels.append(data[2][1])

                # 评估的时候数据也要放到gpu上
                batch_graphs = dgl.batch(graphs).to(self.device)
                batch_hypergraphs = batch_hypergraph(hypergraphs, device=self.device)

                instance_labels = torch.tensor(instance_labels)
                type_labels = torch.tensor(type_labels)

                model.eval()
                start_time = time.time()
                with torch.no_grad():
                    root_logit, type_logit = model(batch_graphs, batch_hypergraphs)

                end_time = time.time()
                time_test = (end_time - start_time)
                test_times.append(time_test)
                top1, top2, top3, top4, top5 = accuracy(root_logit.cpu(), instance_labels,batch_graphs,
                                                                       topk=(1, 2, 3, 4, 5))
                # nan_top1_root, top1, top2, top3, top4, top5 = accuracy(root_logit.cpu(), instance_labels, batch_graphs, topk=(1, 2, 3, 4, 5), eval=True)
                avg_5 = np.mean([top1, top2, top3, top4, top5])
                type_logit = type_logit.detach().cpu()
                pre = precision(type_logit, type_labels, k=5)
                rec = recall(type_logit, type_labels, k=5)
                f1 = f1score(type_logit, type_labels, k=5)

                self.logger.info("Validation Results - Epoch: {}".format(epoch))
                # self.logger.info(
                #     "[Root localization] top1: {:.3%}, top2: {:.3%}, top3: {:.3%}, top4: {:.3%}, top5: {:.3%}, avg@5: {:.3f}, nan_top1_root: \n{}".format(
                #         top1, top2, top3, top4, top5, avg_5, nan_top1_root))
                self.logger.info(
                    "[Root localization] top1: {:.3%}, top2: {:.3%}, top3: {:.3%}, top4: {:.3%}, top5: {:.3%}, avg@5: {:.3f}".format(
                        top1, top2, top3, top4, top5, avg_5))
                self.logger.info(
                    "[Failure type classification] precision: {:.3%}, recall: {:.3%}, f1-score: {:.3%}".format(pre, rec,
                                                                                                               f1))

                self.writer.add_scalar('test/top1', top1, global_step=n_test)
                self.writer.add_scalar('test/top3', top3, global_step=n_test)
                self.writer.add_scalar('test/top5', top5, global_step=n_test)
                self.writer.add_scalar('test/precision', pre, global_step=n_test)
                self.writer.add_scalar('test/recall', rec, global_step=n_test)
                self.writer.add_scalar('test/f1-score', f1, global_step=n_test)
                self.logger.info("Time of test: {:.3f}[s]"
                                 .format(time_test))

                if (avg_5 + f1) > (best_avg + best_f1):
                    best_avg = avg_5
                    best_f1 = f1
                    best_data['top1'] = top1
                    best_data['top2'] = top2
                    best_data['top3'] = top3
                    best_data['top4'] = top4
                    best_data['top5'] = top5
                    best_data['avg_5'] = avg_5
                    best_data['precision'] = pre
                    best_data['recall'] = rec
                    best_data['f1-score'] = f1
                    # best_data['nan_top1_root'] = nan_top1_root
                    state = {
                        'epoch': self.epochs,
                        'model': model.state_dict(),
                        'opt': opt.state_dict(),
                    }
                    torch.save(state, os.path.join(self.writer.log_dir, 'model_best.pth.tar'))

        self.logger.info("Training has finished.")
        # calculate the training time for raw data
        self.logger.debug(f"The average training time per epoch is {np.mean(fit_times) / 2}")
        self.logger.debug(f"The average predict time is {np.mean(test_times)}")
        self.logger.debug(f"The affinity of RCL -> FTI is {np.mean(Z_r2f)}")
        self.logger.debug(f"The affinity of FTI -> RCL is {np.mean(Z_f2r)}")
        for key in best_data.keys():
            self.logger.debug(f'{key}: {best_data[key]}')
            print(f'{key}: {best_data[key]}')
        self.logger.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
