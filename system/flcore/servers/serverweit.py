"""
FedWeIT server with FedAvg-style printing and tracking, while preserving FedWeIT's
original training algorithm (NetModule/TrainModule aggregation and client-side WeIT
updates).

What this adds (vs. your original FedWeIT):
  • RichRoundLogger (same look as FedAvg): per-round header, per-client table, summary panel.
  • Per-client AA@t (average accuracy up to current task) on the GLOBAL test pool.
  • Per-client Average Forgetting (so-far) with metric_average_forgetting.
  • Optional delta-L2 of local updates for quick sanity.
  • Dump a GLOBAL per-task accuracy row ONCE at the end of each task.

Training sequence is kept:
  - select clients → client.train_one_round(..., get_weights(), get_adapts(...))
  - collect updates → aggregate via TrainModule → set_weights(...)
"""

import os
import time
import copy
import shutil
import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from flcore.clients.clientweit import clientWeIT
from flcore.servers.serverbase import Server
from flcore.trainmodel.fedewit_models import NetModule, TrainModule, Logger
from flcore.utils_core.fedweit_utils import *

from flcore.metrics.average_forgetting import metric_average_forgetting
from utils.data_utils import (read_client_data_FCL_cifar100,
                              read_client_data_FCL_imagenet1k, read_client_data_FCL_cifar10)
# Pretty logger (safe if not installed)
try:
    from utils.rich_progress import RichRoundLogger
except Exception:
    RichRoundLogger = None


class FedWeIT(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # Original FedWeIT init
        args.log_dir   = os.path.join(args.output_path, f'logs/{args.dataset}')
        args.state_dir = os.path.join(args.output_path, f'states/{args.dataset}')
        self.Budget: List[float] = []
        self.client_adapts: List[Any] = []
        self.logger = Logger(self.args)
        self.nets   = NetModule(self.args)
        self.trainh = TrainModule(self.args, self.logger, self.nets)
        self.nets.init_state(None)
        self.trainh.init_state(None)
        self.global_weights = self.nets.init_global_weights()

        # Create clients (original)
        self.set_clients(clientWeIT)
        print(f"Number of clients: {len(self.clients)}")
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # FedAvg-style tracking state
        if not hasattr(self, "client_accuracy_matrix"):
            self.client_accuracy_matrix: Dict[int, List[List[float]]] = {}

        if not hasattr(self, "_roundlog") or self._roundlog is None:
            if RichRoundLogger is not None:
                fig_dir = getattr(self.args, "fig_dir", "figures")
                pretty_on = getattr(self.args, "pretty_log", True)
                self._roundlog = RichRoundLogger(self.args, fig_dir=fig_dir, enable=pretty_on)
            else:
                class _Dummy:
                    def start(self, total_rounds): print(f"[Progress] total planned rounds: {total_rounds}")
                    def round_start(self, round_idx, task_id, selected_clients):
                        print(f"\n=== Round {round_idx} | Task {task_id} | Clients {selected_clients} ===")
                    def clients_end(self, round_idx, client_summaries): pass
                    def round_end(self, round_idx, global_metrics, time_cost=None):
                        print(f"[Summary] round={round_idx} time={time_cost:.2f}s")
                    def finish(self): pass
                self._roundlog = _Dummy()

    # ----------------------- helpers -----------------------
    def _cid(self, c, idx):
        return getattr(c, "id", getattr(c, "cid", idx))

    def _flatten_params(self, model: torch.nn.Module) -> Optional[torch.Tensor]:
        try:
            with torch.no_grad():
                return torch.cat([p.detach().float().view(-1).cpu() for p in model.parameters() if p is not None])
        except Exception:
            return None

    def _acc_vec_for_client_from_counts(self, client, cc: np.ndarray, tt: np.ndarray) -> List[float]:
        """
        Build [A_{*,k}]_{k=0..T-1} for THIS client from per-class (cc,tt) counts,
        using the server's robust label resolver for (client, task_idx).
        """
        vec: List[float] = []
        for k in range(self.N_TASKS):
            labels = self._labels_for_client_task(client, k)  # GLOBAL class IDs
            if not labels:
                vec.append(0.0)
                continue
            idx = np.asarray(labels, dtype=np.int64)
            corr = int(cc[idx].sum())
            tot  = int(tt[idx].sum())
            vec.append((corr / tot) if tot > 0 else 0.0)
        return vec

    # ----------------------- FedWeIT training with FedAvg-style logging -----------------------
    def train(self):
        # Optional: clean output (as in your original)
        if os.path.exists("./output_fedweit"):
            shutil.rmtree("./output_fedweit")

        num_tasks     = int(self.args.num_tasks)
        total_rounds  = int(self.global_rounds) * num_tasks
        eval_gap      = int(getattr(self, "eval_gap", 1) or 1)

        self._roundlog.start(total_rounds=total_rounds)

        for task in range(num_tasks):
            print(f"\n================ Current Task: {task} =================")
            self.current_task = task
            torch.cuda.empty_cache()

            # initial task 0: propagate available labels (same as other servers)
            if task == 0:
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.clients:
                    available_labels |= set(u.classes_so_far)
                    available_labels_current |= set(u.current_labels)
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
            else:
                # Load next-task data for each client (original)
                for i in range(len(self.clients)):
                    if self.args.partition_options == 'tuan':
                        from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
                        if self.args.dataset == 'IMAGENET1k':
                            train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                        elif self.args.dataset == 'CIFAR100':
                            train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                        else:
                            raise NotImplementedError("Not supported dataset")
                    elif self.args.partition_options == 'hetero':
                        from utils.data_utils_mine import read_client_data_FCL_cifar10, read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
                        if self.args.dataset == 'IMAGENET1k':
                            train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True,
                                                                                    seed = self.args.seed, alpha = self.args.alpha,
                                                                                    total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                        elif self.args.dataset == 'CIFAR100':
                            train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True,
                                                                                seed = self.args.seed, alpha = self.args.alpha,
                                                                                total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                        elif self.args.dataset == 'CIFAR10':
                            train_data, label_info = read_client_data_FCL_cifar10(i, task=task, classes_per_task=self.args.cpt, count_labels=True,
                                                                                seed = self.args.seed, alpha = self.args.alpha,
                                                                                total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                        else:
                            raise NotImplementedError("Not supported dataset")
                    self.clients[i].next_task(train_data, label_info)

                # Update availability trackers
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels
                for u in self.clients:
                    available_labels |= set(u.classes_so_far)
                    available_labels_current |= set(u.current_labels)
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            # ---------------- rounds inside task ----------------
            for i in range(self.global_rounds):
                glob_iter  = i + self.global_rounds * task
                disp_round = glob_iter + 1
                self._round_tag = glob_iter
                s_t = time.time()

                self.updates = []
                self.curr_round   = glob_iter + 1
                self.is_last_round = (i == 0)  # keep original behavior

                if self.is_last_round:
                    self.client_adapts = []

                # (1) select clients
                self.selected_clients = self.select_clients()
                sel_ids = [self._cid(c, j) for j, c in enumerate(self.selected_clients)]

                # header
                self._roundlog.round_start(round_idx=disp_round, task_id=task, selected_clients=sel_ids)

                # (2) optional global eval (to match FedAvg UX)
                if i % eval_gap == 0 and hasattr(self, "eval"):
                    try:
                        self.eval(task=task, glob_iter=glob_iter, flag="global")
                    except TypeError:
                        try:
                            self.eval()
                        except Exception:
                            pass

                # Build per-class counts ONCE for this eval round (AA/forgetting)
                try:
                    cc, tt = self._get_or_build_global_counts(self._round_tag)
                except Exception:
                    cc, tt, _K = self._compute_global_per_class_counts(self.global_model)

                # (3) local training for selected clients + metrics capture
                client_summaries: List[Dict[str, Any]] = []
                for client in self.selected_clients:
                    # Each client runs one round with current weights and optional adapts
                    t0_cli = time.time()
                    before = self._flatten_params(client.model)

                    update = client.train_one_round(
                        client.id, glob_iter, True,
                        self.get_weights(),
                        self.get_adapts(glob_iter=glob_iter)
                    )
                    if update is not None:
                        self.updates.append(update)
                    if self.is_last_round:
                        self.client_adapts.append(client.get_adaptives())

                    # metrics
                    delta_l2 = None
                    try:
                        after = self._flatten_params(client.model)
                        if before is not None and after is not None and after.shape == before.shape:
                            delta_l2 = float(torch.norm(after - before, p=2))
                    except Exception:
                        pass

                    train_loss = None
                    try:
                        tr_sum, tr_n = client.train_metrics(task=task)
                        train_loss = (tr_sum / max(1, tr_n)) if tr_n else None
                    except Exception:
                        pass

                    # AA up to task and forgetting (like FedAvg)
                    acc_vec = self._acc_vec_for_client_from_counts(client, cc, tt)
                    self.client_accuracy_matrix.setdefault(client.id, []).append(acc_vec)
                    aa_pct = float(100.0 * (np.mean(acc_vec[:task + 1]) if (task + 1) > 0 else 0.0))
                    try:
                        cf = metric_average_forgetting(int(task % self.N_TASKS), self.client_accuracy_matrix[client.id])
                        forg_pct = float(100.0 * cf)
                    except Exception:
                        forg_pct = None

                    client_summaries.append({
                        "client": client.id,
                        "loss":   train_loss,
                        "acc":    aa_pct,      # %
                        "forg":   forg_pct,    # %
                        "time":   time.time() - t0_cli,
                        "samples": getattr(client, "train_samples", None),
                        "delta_l2": delta_l2,
                    })

                # table
                self._roundlog.clients_end(round_idx=disp_round, client_summaries=client_summaries)

                # (4) aggregate FedWeIT updates
                aggr = self.trainh.aggregate(self.updates)
                self.set_weights(aggr)

                # (5) end-of-round summary (plus optional quick global metrics)
                elapsed = time.time() - s_t
                self.Budget.append(elapsed)
                g_metrics = {}
                if i % eval_gap == 0 and hasattr(self, "test"):
                    try:
                        m = self.test()
                        if isinstance(m, dict):
                            g_metrics = {
                                "test_loss": m.get("test_loss", m.get("loss")),
                                "test_acc":  m.get("test_acc",  m.get("acc")),
                            }
                    except Exception:
                        pass
                self._roundlog.round_end(round_idx=disp_round, global_metrics=g_metrics, time_cost=elapsed)

        # finish progress
        self._roundlog.finish()

    # --------------------- WeIT accessors (kept) ---------------------
    def get_weights(self):
        return self.global_weights

    def set_weights(self, weights):
        self.global_weights = weights

    def get_adapts(self, glob_iter):
        # original logic
        if glob_iter % self.global_rounds == 1 and not glob_iter == 1:
            from_kb = []
            for lid, shape in enumerate(self.nets.shapes):
                shape = np.concatenate([self.nets.shapes[lid], [int(round(self.args.num_clients * self.join_ratio))]], axis=0)
                from_kb_l = np.zeros(shape)
                for cid, ca in enumerate(self.client_adapts):
                    try:
                        if len(shape) == 5:
                            from_kb_l[:, :, :, :, cid] = ca[lid]
                        else:
                            from_kb_l[:, :, cid] = ca[lid]
                    except Exception:
                        import pdb; pdb.set_trace()
                from_kb.append(from_kb_l)
            return from_kb
        else:
            return None

    # --------------------- Client creation (kept) ---------------------
    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            if self.args.partition_options == 'tuan':
                from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
                if self.args.dataset == 'IMAGENET1k':
                    train_data, label_info = read_client_data_FCL_imagenet1k(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
                elif self.args.dataset == 'CIFAR100':
                    train_data, label_info = read_client_data_FCL_cifar100(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
                else:
                    raise NotImplementedError("Not supported dataset")
            elif self.args.partition_options == 'hetero':
                if self.args.dataset == 'IMAGENET1k':
                    train_data, label_info = read_client_data_FCL_imagenet1k(i, task=0, classes_per_task=self.args.cpt, count_labels=True,
                                                                            seed = self.args.seed, alpha = self.args.alpha,
                                                                            total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                elif self.args.dataset == 'CIFAR100':
                    train_data, label_info = read_client_data_FCL_cifar100(i, task=0, classes_per_task=self.args.cpt, count_labels=True,
                                                                        seed = self.args.seed, alpha = self.args.alpha,
                                                                        total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                elif self.args.dataset == 'CIFAR10':
                    train_data, label_info = read_client_data_FCL_cifar10(i, task=0, classes_per_task=self.args.cpt, count_labels=True,
                                                                        seed = self.args.seed, alpha = self.args.alpha,
                                                                        total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                else:
                    raise NotImplementedError("Not supported dataset")

            client = clientObj(self.args, id=i, train_data=train_data, initial_weights=self.global_weights)
            self.clients.append(client)

            # update classes so far & current labels
            client.classes_so_far.extend(label_info['labels'])
            client.current_labels.extend(label_info['labels'])