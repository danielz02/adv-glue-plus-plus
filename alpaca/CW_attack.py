import sys
import torch
import copy
import numpy as np
from torch import optim
from tqdm import tqdm

from model import ZeroShotLlamaForSemAttack
from util import get_args


class CarliniL2:

    def __init__(self, args, logger, targeted=True, search_steps=None, max_steps=None, device=None, debug=False,
                 num_classes=3):
        logger.info(("const confidence lr:", args.const, args.confidence, args.lr))
        self.args = args
        self.debug = debug
        self.targeted = targeted
        self.num_classes = num_classes
        self.confidence = args.confidence  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        self.initial_const = args.const  # bumped up from default of .01 in reference code
        self.binary_search_steps = search_steps or 1
        self.repeat = self.binary_search_steps >= 10
        self.max_steps = max_steps or args.max_steps
        self.abort_early = True
        self.device = device if device is not None else torch.device("cpu")
        self.mask = None
        self.batch_info = None
        self.wv = None
        self.input_dict = None
        self.seq = None
        self.init_rand = False  # an experiment, does a random starting point help?
        self.best_sent = None
        self.o_best_sent = None
        self.tokenizer = None

    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            # if self.targeted:
            #     output[target] -= self.confidence
            # else:
            #     output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target
        else:
            return output != target

    def _compare_untargeted(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            # if self.targeted:
            #     output[target] -= self.confidence
            # else:
            #     output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target + 1 or output == target - 1
        else:
            return output != target

    def _loss(self, output, target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = torch.sum(scale_const * loss1)
        loss2 = dist.sum()
        loss = loss1 + loss2
        return loss

    def _optimize(self, optimizer, model: ZeroShotLlamaForSemAttack, input_var, modifier_var, target_var,
                  scale_const_var, input_token=None):
        # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max

        batch_adv_sent = []
        if self.mask is None:
            # not word-level attack
            input_adv = modifier_var + input_var
            output = model(self.input_dict, perturbed=input_adv)
            input_adv = model.get_embedding()  # FIXME: What is get_embedding()?
            input_var = input_token
            seqback = model.get_seqback()  # FIXME: What is get_seqback()?
            batch_adv_sent = seqback.adv_sent.copy()
            seqback.adv_sent = []
            # input_adv = self.itereated_var = modifier_var + self.itereated_var
        else:
            # word level attack
            input_adv = modifier_var * self.mask + self.itereated_var
            # input_adv = modifier_var * self.mask + input_var
            new_word_list = []
            add_start = self.batch_info['add_start'][0]
            add_end = self.batch_info['add_end'][0]
            for j in range(add_start, add_end):
                # print(self.wv[self.seq[0][j].item()])
                # if self.seq[0][j].item() not in self.wv.keys():

                similar_wv = model.get_input_embedding_vector(
                    torch.tensor(self.wv[self.seq[j].item()], dtype=torch.long).to(self.device)
                )
                new_placeholder = input_adv[j].data
                temp_place = new_placeholder.expand_as(similar_wv)
                new_dist = torch.norm(temp_place - similar_wv, 2, -1)  # 2范数距离，一个字一个float
                _, new_word = torch.min(new_dist, 0)
                # print(j, "new_dist", new_dist, "new_word", new_word)
                new_word_list.append(new_word.item())
                # input_adv.data[j, i] = self.wv[new_word.item()].data
                input_adv.data[j] = similar_wv[new_word.item()].data
                del temp_place
            batch_adv_sent.append(new_word_list)

            output = model(input_dict=self.input_dict, perturbed=input_adv)["logits"]

        def reduce_sum(x, keepdim=True):
            # silly PyTorch, when will you get proper reducing sums/means?
            for a in reversed(range(1, x.dim())):
                x = x.sum(a, keepdim=keepdim)
            return x

        def l1_dist(x, y, keepdim=True):
            d = torch.abs(x - y)
            return reduce_sum(d, keepdim=keepdim)

        def l2_dist(x, y, keepdim=True):
            d = (x - y) ** 2
            return reduce_sum(d, keepdim=keepdim)

        # distance to the original input_embedding data
        if self.args.l1:
            dist = l1_dist(input_adv.unsqueeze(0), input_var.unsqueeze(0), keepdim=False)
        else:
            # Add batch dimension
            dist = l2_dist(input_adv.unsqueeze(0), input_var.unsqueeze(0), keepdim=False)
        loss = self._loss(output, target_var, dist, scale_const_var)
        optimizer.zero_grad()
        if input_token is None:
            loss.backward()
        else:
            loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_([modifier_var], self.args.clip)  # 0.5
        optimizer.step()
        # modifier_var.data -= 2 * modifier_var.grad.data
        # modifier_var.grad.data.zero_()

        loss_np = loss.item()
        dist_np = dist.detach().cpu().numpy()
        output_np = output.data.cpu().numpy()
        input_adv_np = input_adv.data.cpu().numpy()
        return loss_np, dist_np, output_np, input_adv_np, batch_adv_sent

    def run(self, model, input_embedding, target, input_dict, batch_idx=0, batch_size=None, input_token=None):
        self.input_dict = copy.deepcopy(input_dict)
        self.input_dict['input_ids'] = None
        # set the lower and upper bounds accordingly
        lower_bound = 0
        scale_const = self.initial_const
        upper_bound = 1e10

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = 1e10
        o_best_score = -1
        o_best_logits = None
        if input_token is None:
            best_attack = input_embedding.cpu().detach().numpy()
            o_best_attack = input_embedding.cpu().detach().numpy()
        else:
            best_attack = input_token.cpu().detach().numpy()
            o_best_attack = input_token.cpu().detach().numpy()
        self.o_best_sent = {}
        self.best_sent = {}

        # TODO: Double check copy construction
        # setup input_embedding (image) variable, clamp/scale as necessary
        input_var = input_embedding.clone().detach().requires_grad_(False)
        self.itereated_var = input_var.clone()
        # setup the target variable, we need it to be in one-hot form for the loss function
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if self.device:
            target_onehot = target_onehot.to(self.device)

        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = target_onehot.clone().detach().requires_grad_(False)

        # setup the modifier variable, this is the variable we are optimizing over
        modifier = torch.zeros_like(input_var).float()
        if self.device:
            modifier = modifier.to(self.device)
        modifier_var = modifier.clone().detach().requires_grad_(True)

        optimizer = optim.Adam([modifier_var], lr=self.args.lr)  # 0.1

        for search_step in range(self.binary_search_steps):
            best_l2 = 1e10
            best_score = -1
            best_logits = {}
            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor = torch.tensor([scale_const]).float()
            if self.device:
                scale_const_tensor = scale_const_tensor.to(self.device)
            scale_const_var = scale_const_tensor.clone().detach().requires_grad_(False)

            for step in tqdm(range(self.max_steps)):
                # perform the attack
                if self.mask is None:
                    if self.args.decreasing_temp:
                        cur_temp = self.args.temp - (self.args.temp - 0.1) / (self.max_steps - 1) * step
                        model.set_temp(cur_temp)
                        if self.args.debug_cw:
                            print("temp:", cur_temp)
                    else:
                        model.set_temp(self.args.temp)
                # modifier_var = modifier.clone().detach().requires_grad_(True)
                # output 是攻击后的model的test输出  adv_img是输出的词向量矩阵， adv_sents是字的下标组成的list
                loss, dist, output, adv_img, adv_sents = self._optimize(
                    optimizer,
                    model,
                    input_var,
                    modifier_var,
                    target_var,
                    scale_const_var,
                    input_token
                )
                # print("Adv Sentence", self.tokenizer.decode(adv_sents[0]))

                target_label = target.item()
                output_logits = output.reshape(-1)  # Removing batch dimension...
                output_label = np.argmax(output_logits)
                adv_sents = adv_sents[0]
                di = dist[0]
                if self.debug:
                    # if step % 100 == 0:
                        print(
                            'dist: {0:.5f}, output: {1:>3}, {2:5.3}, target {3:>3}'
                            .format(di, output_label, output_logits[output_label], target_label)
                        )
                if di < best_l2 and self._compare_untargeted(output_logits, target_label):
                    if self.debug:
                        print('best step,  prev dist: {0:.5f}, new dist: {1:.5f}'.format(best_l2, di))
                    best_l2 = di
                    best_score = output_label
                    best_logits = output_logits
                    best_attack = adv_img
                    self.best_sent = adv_sents
                if di < o_best_l2 and self._compare(output_logits, target_label):
                    if self.debug:
                        print('best total, prev dist: {0:.5f}, new dist: {1:.5f}'.format(o_best_l2, di))
                    o_best_l2 = di
                    o_best_score = output_label
                    o_best_logits = output_logits
                    o_best_attack = adv_img
                    self.o_best_sent = adv_sents

                sys.stdout.flush()
                # end inner step loop

            # adjust the constants
            batch_failure = 0
            batch_success = 0

            if self._compare(o_best_score, target) and o_best_score != -1:
                batch_success += 1
            elif self._compare_untargeted(best_score, target) and best_score != -1:
                o_best_l2 = best_l2
                o_best_score = best_score
                o_best_attack = best_attack
                self.o_best_sent = self.best_sent
                batch_success += 1
            else:
                batch_failure += 1

            # print('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()
            # end outer search loop

        return o_best_attack
