#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified KGEModel with:
  - TransE, DistMult, ComplEx, RotatE, pRotatE (baselines)
  - RotateCT     : rotation around relation-specific centers (b_r)
  - MRotatE      : entity rotation + relation rotation (about origin)
  - MRotatECT    : entity rotation + relation rotation AROUND b_r (RotateCT inside)

Both RotateCT, MRotatE, and MRotatECT require complex entity embeddings:
use CLI flag `-de / --double_entity_embedding`.
"""

from __future__ import absolute_import, division, print_function

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

# Provided in your project
from dataloader import TestDataset


class KGEModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        nentity: int,
        nrelation: int,
        hidden_dim: int,
        gamma: float,
        double_entity_embedding: bool = False,
        double_relation_embedding: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        # Margin (kept as nn.Parameter for checkpoint compatibility)
        self.gamma = nn.Parameter(torch.tensor([gamma], dtype=torch.float), requires_grad=False)

        # Used for init and phase scaling
        self.embedding_range = nn.Parameter(
            torch.tensor([(self.gamma.item() + self.epsilon) / hidden_dim], dtype=torch.float),
            requires_grad=False,
        )

        # Complex entities => concatenated (Re | Im)
        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        # Only ComplEx uses complex relations
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        # Base embeddings
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(self.entity_embedding, -self.embedding_range.item(), self.embedding_range.item())

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(self.relation_embedding, -self.embedding_range.item(), self.embedding_range.item())

        # pRotatE extra param
        if model_name == "pRotatE":
            self.modulus = nn.Parameter(torch.tensor([[0.5 * self.embedding_range.item()]], dtype=torch.float))

        # ---------------- Supported names (added MRotatECT) ----------------
        supported = ["TransE", "DistMult", "ComplEx", "RotatE", "pRotatE", "RotateCT", "MRotatE", "MRotatECT"]
        if model_name not in supported:
            raise ValueError(f"model {model_name} not supported; choose from {supported}")

        # Constraints
        if model_name == "RotatE" and (not double_entity_embedding or double_relation_embedding):
            raise ValueError("RotatE should use --double_entity_embedding and NOT --double_relation_embedding")
        if model_name == "ComplEx" and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError("ComplEx should use --double_entity_embedding and --double_relation_embedding")

        # Relation-specific center b_r (complex) needed for RotateCT and MRotatECT
        if model_name in ["RotateCT", "MRotatECT"]:
            if not double_entity_embedding:
                raise ValueError(f"{model_name} requires --double_entity_embedding (complex entities)")
            self.relation_center_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(self.relation_center_embedding, -self.embedding_range.item(), self.embedding_range.item())

        # Per-entity phase angles φ_e needed for MRotatE and MRotatECT
        if model_name in ["MRotatE", "MRotatECT"]:
            if not double_entity_embedding:
                raise ValueError(f"{model_name} requires --double_entity_embedding (complex entities)")
            self.entity_phase_embedding = nn.Parameter(torch.zeros(nentity, hidden_dim))
            nn.init.uniform_(self.entity_phase_embedding, -self.embedding_range.item(), self.embedding_range.item())

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, sample: torch.LongTensor, mode: str = "single") -> torch.Tensor:
        """
        Modes:
          - 'single'     : sample is [B, 3]
          - 'head-batch' : sample is (positive_part [B,3], corrupted_heads [B, K])
          - 'tail-batch' : sample is (positive_part [B,3], corrupted_tails [B, K])
        """
        if mode == "single":
            head = torch.index_select(self.entity_embedding, 0, sample[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, 0, sample[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, 0, sample[:, 2]).unsqueeze(1)

            if self.model_name in ["RotateCT", "MRotatECT"]:
                relation_center = torch.index_select(self.relation_center_embedding, 0, sample[:, 1]).unsqueeze(1)
            if self.model_name in ["MRotatE", "MRotatECT"]:
                head_phase = torch.index_select(self.entity_phase_embedding, 0, sample[:, 0]).unsqueeze(1)
                tail_phase = torch.index_select(self.entity_phase_embedding, 0, sample[:, 2]).unsqueeze(1)

        elif mode == "head-batch":
            tail_part, head_part = sample
            head = torch.index_select(self.entity_embedding, 0, head_part.view(-1)).view(
                head_part.size(0), head_part.size(1), -1
            )
            relation = torch.index_select(self.relation_embedding, 0, tail_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, 0, tail_part[:, 2]).unsqueeze(1)

            if self.model_name in ["RotateCT", "MRotatECT"]:
                relation_center = torch.index_select(self.relation_center_embedding, 0, tail_part[:, 1]).unsqueeze(1)
            if self.model_name in ["MRotatE", "MRotatECT"]:
                head_phase = torch.index_select(self.entity_phase_embedding, 0, head_part.view(-1)).view(
                    head_part.size(0), head_part.size(1), -1
                )
                tail_phase = torch.index_select(self.entity_phase_embedding, 0, tail_part[:, 2]).unsqueeze(1)

        elif mode == "tail-batch":
            head_part, tail_part = sample
            head = torch.index_select(self.entity_embedding, 0, head_part[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, 0, head_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, 0, tail_part.view(-1)).view(
                tail_part.size(0), tail_part.size(1), -1
            )

            if self.model_name in ["RotateCT", "MRotatECT"]:
                relation_center = torch.index_select(self.relation_center_embedding, 0, head_part[:, 1]).unsqueeze(1)
            if self.model_name in ["MRotatE", "MRotatECT"]:
                head_phase = torch.index_select(self.entity_phase_embedding, 0, head_part[:, 0]).unsqueeze(1)
                tail_phase = torch.index_select(self.entity_phase_embedding, 0, tail_part.view(-1)).view(
                    tail_part.size(0), tail_part.size(1), -1
                )
        else:
            raise ValueError(f"mode {mode} not supported")

        # Dispatch
        if self.model_name == "RotateCT":
            return self.RotateCT(head, relation, tail, relation_center, mode)
        if self.model_name == "MRotatE":
            return self.MRotatE(head, relation, tail, head_phase, tail_phase, mode)
        if self.model_name == "MRotatECT":
            return self.MRotatECT(head, relation, tail, head_phase, tail_phase, relation_center, mode)

        model_func = {
            "TransE": self.TransE,
            "DistMult": self.DistMult,
            "ComplEx": self.ComplEx,
            "RotatE": self.RotatE,
            "pRotatE": self.pRotatE,
        }
        return model_func[self.model_name](head, relation, tail, mode)

    # ---------------------------------------------------------------------
    # Baselines
    # ---------------------------------------------------------------------
    def TransE(self, head, relation, tail, mode):
        score = head + (relation - tail) if mode == "head-batch" else (head + relation) - tail
        return self.gamma.item() - torch.norm(score, p=1, dim=2)

    def DistMult(self, head, relation, tail, mode):
        score = head * (relation * tail) if mode == "head-batch" else (head * relation) * tail
        return score.sum(dim=2)

    def ComplEx(self, head, relation, tail, mode):
        re_h, im_h = torch.chunk(head, 2, dim=2)
        re_r, im_r = torch.chunk(relation, 2, dim=2)
        re_t, im_t = torch.chunk(tail, 2, dim=2)
        if mode == "head-batch":
            re_s = re_r * re_t + im_r * im_t
            im_s = re_r * im_t - im_r * re_t
            score = re_h * re_s + im_h * im_s
        else:
            re_s = re_h * re_r - im_h * im_r
            im_s = re_h * im_r + im_h * re_r
            score = re_s * re_t + im_s * im_t
        return score.sum(dim=2)

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        re_h, im_h = torch.chunk(head, 2, dim=2)
        re_t, im_t = torch.chunk(tail, 2, dim=2)
        phase_r = relation / (self.embedding_range.item() / pi)
        re_r, im_r = torch.cos(phase_r), torch.sin(phase_r)
        if mode == "head-batch":
            re_s = re_r * re_t + im_r * im_t
            im_s = re_r * im_t - im_r * re_t
            re_s = re_s - re_h
            im_s = im_s - im_h
        else:
            re_s = re_h * re_r - im_h * im_r
            im_s = re_h * im_r + im_h * re_r
            re_s = re_s - re_t
            im_s = im_s - im_t
        score = torch.stack([re_s, im_s], dim=0).norm(dim=0)
        return self.gamma.item() - score.sum(dim=2)

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        ph_h = head / (self.embedding_range.item() / pi)
        ph_r = relation / (self.embedding_range.item() / pi)
        ph_t = tail / (self.embedding_range.item() / pi)
        score = ph_h + (ph_r - ph_t) if mode == "head-batch" else (ph_h + ph_r) - ph_t
        score = torch.sin(score).abs()
        return self.gamma.item() - score.sum(dim=2) * self.modulus

    # ---------------------------------------------------------------------
    # RotateCT (about relation center)
    # ---------------------------------------------------------------------
    def RotateCT(self, head, relation, tail, relation_center, mode):
        pi = 3.14159265358979323846
        re_h, im_h = torch.chunk(head, 2, dim=2)
        re_t, im_t = torch.chunk(tail, 2, dim=2)
        re_b, im_b = torch.chunk(relation_center, 2, dim=2)
        phase_r = relation / (self.embedding_range.item() / pi)
        re_r, im_r = torch.cos(phase_r), torch.sin(phase_r)
        if mode == "head-batch":
            re_rot = re_r * (re_t - re_b) + im_r * (im_t - im_b)
            im_rot = re_r * (im_t - im_b) - im_r * (re_t - re_b)
            re_s = re_rot - (re_h - re_b)
            im_s = im_rot - (im_h - im_b)
        else:
            re_rot = (re_h - re_b) * re_r - (im_h - im_b) * im_r
            im_rot = (re_h - re_b) * im_r + (im_h - im_b) * re_r
            re_s = re_rot - (re_t - re_b)
            im_s = im_rot - (im_t - im_b)
        score = torch.stack([re_s, im_s], dim=0).norm(dim=0)
        return self.gamma.item() - score.sum(dim=2)

    # ---------------------------------------------------------------------
    # MRotatE (entity rotation + relation rotation about origin)
    # ---------------------------------------------------------------------
    def MRotatE(self, head, relation, tail, head_phase, tail_phase, mode):
        pi = 3.14159265358979323846
        re_h, im_h = torch.chunk(head, 2, dim=2)
        re_t, im_t = torch.chunk(tail, 2, dim=2)
        ph_r = relation / (self.embedding_range.item() / pi)
        re_r, im_r = torch.cos(ph_r), torch.sin(ph_r)
        ph_h = head_phase / (self.embedding_range.item() / pi)
        ph_t = tail_phase / (self.embedding_range.item() / pi)
        re_eh, im_eh = torch.cos(ph_h), torch.sin(ph_h)
        re_et, im_et = torch.cos(ph_t), torch.sin(ph_t)
        Eh_re = re_eh * re_h - im_eh * im_h
        Eh_im = re_eh * im_h + im_eh * re_h
        Et_re = re_et * re_t - im_et * im_t
        Et_im = re_et * im_t + im_et * re_t
        Rh_re = Eh_re * re_r - Eh_im * im_r
        Rh_im = Eh_re * im_r + Eh_im * re_r
        if mode == "head-batch":
            re_s = (re_r * Et_re + im_r * Et_im) - Eh_re
            im_s = (re_r * Et_im - im_r * Et_re) - Eh_im
        else:
            re_s = Rh_re - Et_re
            im_s = Rh_im - Et_im
        score = torch.stack([re_s, im_s], dim=0).norm(dim=0)
        return self.gamma.item() - score.sum(dim=2)

    # ---------------------------------------------------------------------
    # NEW: MRotatECT (entity rotation + RotateCT around b_r)
    # ---------------------------------------------------------------------
    def MRotatECT(self, head, relation, tail, head_phase, tail_phase, relation_center, mode):
        """
        MRotatECT:
          - Entity rotation: E(x) = R(φ_x) ⊙ x
          - Relation rotation AROUND center b_r (RotateCT):
                (E(h) - b_r) ⊙ R(θ_r)  ≈  (E(t) - b_r)
        This combines multiplicity capacity (entity rotation) with
        non-commutative composition (rotation about b_r).
        """
        pi = 3.14159265358979323846

        # Split complex entity parts
        re_h, im_h = torch.chunk(head, 2, dim=2)
        re_t, im_t = torch.chunk(tail, 2, dim=2)
        re_b, im_b = torch.chunk(relation_center, 2, dim=2)

        # Convert angles to unit complex rotations
        ph_r = relation / (self.embedding_range.item() / pi)   # θ_r
        re_r, im_r = torch.cos(ph_r), torch.sin(ph_r)

        ph_h = head_phase / (self.embedding_range.item() / pi) # φ_h
        ph_t = tail_phase / (self.embedding_range.item() / pi) # φ_t
        re_eh, im_eh = torch.cos(ph_h), torch.sin(ph_h)
        re_et, im_et = torch.cos(ph_t), torch.sin(ph_t)

        # 1) Entity rotations: E(h), E(t)
        Eh_re = re_eh * re_h - im_eh * im_h
        Eh_im = re_eh * im_h + im_eh * re_h
        Et_re = re_et * re_t - im_et * im_t
        Et_im = re_et * im_t + im_et * re_t

        # 2) Rotate AROUND b_r (like RotateCT)
        if mode == "head-batch":
            # rotate (E(t) - b) then compare with (E(h) - b)
            re_rot = re_r * (Et_re - re_b) + im_r * (Et_im - im_b)
            im_rot = re_r * (Et_im - im_b) - im_r * (Et_re - re_b)
            re_s = re_rot - (Eh_re - re_b)
            im_s = im_rot - (Eh_im - im_b)
        else:
            # rotate (E(h) - b) then compare with (E(t) - b)
            re_rot = (Eh_re - re_b) * re_r - (Eh_im - im_b) * im_r
            im_rot = (Eh_re - re_b) * im_r + (Eh_im - im_b) * re_r
            re_s = re_rot - (Et_re - re_b)
            im_s = im_rot - (Et_im - im_b)

        score = torch.stack([re_s, im_s], dim=0).norm(dim=0)
        return self.gamma.item() - score.sum(dim=2)

    # ---------------------------------------------------------------------
    # Train / Test helpers (same API as original)
    # ---------------------------------------------------------------------
    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        '''
    A single train step. Apply back-propagation and return the loss
    :param model: The KGEModel instance
    :param optimizer: The optimizer (e.g., Adam)
    :param train_iterator: BidirectionalOneShotIterator for training data
    :param args: Parsed arguments from argparse
    :param step: Current training step (used for temperature decay)
    '''

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            # Calculate dynamic temperature based on step
            current_temperature = args.initial_adversarial_temperature * (1 - args.decay_rate) ** step
            current_temperature = max(current_temperature, 0.01)  # Prevent temperature from becoming too small
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * current_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
            'current_temperature': current_temperature if args.negative_adversarial_sampling else 0.0
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
