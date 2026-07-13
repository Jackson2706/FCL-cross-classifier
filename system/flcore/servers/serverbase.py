import copy
import csv
import gc
import json
import os
import random
import shutil
import statistics
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from flcore.metrics.accounting import (
    CommunicationAccountant,
    ServerComputeTimer,
    estimate_model_bytes,
)
from flcore.metrics.average_anytime_accuracy import metric_average_anytime_accuracy
from flcore.metrics.average_forgetting import metric_average_forgetting
from flcore.metrics.backward_transfer import metric_backward_transfer
from flcore.scheduler.task_scheduler import ASYNC_CONFIG_DEFAULTS
from flcore.scheduler.consolidation import CONSOLIDATION_CONFIG_DEFAULTS
from flcore.reliability.scorer import RELIABILITY_CONFIG_DEFAULTS
from flcore.boundary.attacks import BOUNDARY_CONFIG_DEFAULTS
from flcore.robustness.scenarios import ROBUSTNESS_CONFIG_DEFAULTS
from flcore.hetero.metrics import (
    summarize_distillation_communication,
    summarize_model_type_accuracy,
)
from flcore.hetero.model_factory import (
    build_client_model,
    resolve_heterogeneity_config,
)
from utils.data_utils import *

import wandb


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.heterogeneity_config = resolve_heterogeneity_config(args)
        if self.heterogeneity_config.enabled:
            self.global_model = build_client_model(
                self.heterogeneity_config.server_model, args
            )
        else:
            self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_threthold = args.time_threthold
        self.offlog = args.offlog
        self.save_folder = f"{args.out_folder}/{args.dataset}_{args.algorithm}_{args.model_str}_{args.optimizer}_lr{args.local_learning_rate}_{args.note}" if args.note else f"{args.out_folder}/{args.dataset}_{args.algorithm}_{args.model_str}_{args.optimizer}_lr{args.local_learning_rate}"
        if self.offlog:    
            if os.path.exists(self.save_folder):
                shutil.rmtree(self.save_folder)
            os.makedirs(self.save_folder, exist_ok=True)

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate

        self.global_accuracy_matrix = []
        self.local_accuracy_matrix = []
        self.communication_accountant = CommunicationAccountant()
        self.server_compute_timer = ServerComputeTimer()
        self.server_distillation_timer = ServerComputeTimer()
        self.hetero_communication_records = {}
        self.heterogeneity_metrics = None

        self.async_config = {
            key: getattr(args, key, default)
            for key, default in {
                **ASYNC_CONFIG_DEFAULTS,
                **CONSOLIDATION_CONFIG_DEFAULTS,
            }.items()
        }
        self.reliability_config = {
            key: getattr(args, key, default)
            for key, default in RELIABILITY_CONFIG_DEFAULTS.items()
        }
        self.boundary_config = {
            key: getattr(args, key, default)
            for key, default in BOUNDARY_CONFIG_DEFAULTS.items()
        }
        self.robustness_config = {
            key: getattr(args, key, default)
            for key, default in ROBUSTNESS_CONFIG_DEFAULTS.items()
        }

        self.num_tasks = self._resolve_num_tasks(args)
        # Compatibility alias for servers not yet migrated to the canonical name.
        self.N_TASKS = self.num_tasks

        # FCL
        self.task_dict = {}
        self.current_task = 0

        self.angle_value = 0
        self.grads_angle_value = 0
        self.distance_value = 0
        self.norm_value = 0

        self.file_name = f"{self.args.algorithm}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self._write_resolved_config()

    @staticmethod
    def _resolve_num_tasks(args):
        """Return the canonical total task count used by training and metrics.

        ``args.num_tasks`` is authoritative when present.  Legacy configurations
        without it fall back to the historical dataset default (five for CIFAR-10,
        CIFAR-100, and ImageNet1k), then to ``num_classes // cpt`` when available.
        Accuracy matrices, forgetting, BWT, and AAA must all use this same count.
        """

        configured = getattr(args, "num_tasks", None)
        if configured is not None:
            total = int(configured)
        elif getattr(args, "dataset", None) in {"CIFAR10", "CIFAR100", "IMAGENET1k"}:
            total = 5
        elif getattr(args, "cpt", 0):
            total = int(args.num_classes) // int(args.cpt)
        else:
            raise ValueError("num_tasks is required for this dataset")
        if total <= 0:
            raise ValueError("num_tasks must be positive")
        return total

    def _write_resolved_config(self):
        """Persist the JSON-serializable portion of args for reproducibility."""
        if not self.offlog:
            return
        try:
            resolved = {}
            for key, value in vars(self.args).items():
                if key == "model":
                    continue
                try:
                    json.dumps(value, allow_nan=False)
                except (TypeError, ValueError):
                    continue
                resolved[key] = value

            # Old JSON configs omit these fields; persist their effective defaults.
            resolved.update(self.async_config)
            resolved.update(self.reliability_config)
            resolved.update(self.boundary_config)
            resolved.update(self.robustness_config)
            if self.heterogeneity_config.enabled:
                resolved.update({
                    "model_heterogeneity": True,
                    "client_model_pool": self.heterogeneity_config.client_model_pool,
                    "server_model": self.heterogeneity_config.server_model,
                    "aggregation_mode": self.heterogeneity_config.aggregation_mode,
                    "client_distill_steps": self.heterogeneity_config.client_distill_steps,
                    "distill_transfer_size": self.heterogeneity_config.distill_transfer_size,
                })
            resolved.setdefault(
                "generator_distillation",
                bool(getattr(self.args, "generator_distillation", False)),
            )
            resolved.setdefault(
                "kd_weight",
                float(getattr(self.args, "kd_weight", getattr(self.args, "kd", 0.5))),
            )
            resolved.setdefault(
                "generator_grad_clip",
                float(getattr(self.args, "generator_grad_clip", 10.0)),
            )
            resolved.setdefault("seed", int(getattr(self.args, "seed", 0)))

            config_path = os.path.join(self.save_folder, "resolved_config.json")
            with open(config_path, mode="w") as file:
                json.dump(resolved, file, indent=2, sort_keys=True, allow_nan=False)
                file.write("\n")
        except Exception:
            pass

    @staticmethod
    def _summarize_accuracy_matrix(accuracy_matrix):
        if not accuracy_matrix:
            return {
                "average_accuracy": None,
                "forgetting": None,
                "backward_transfer": None,
                "average_anytime_accuracy": None,
            }

        try:
            stage = len(accuracy_matrix) - 1
            latest_row = accuracy_matrix[-1]
            seen_values = [
                float(value)
                for value in latest_row[: stage + 1]
                if np.isfinite(value)
            ]
            average_accuracy = float(np.mean(seen_values)) if seen_values else None
            forgetting = float(metric_average_forgetting(stage, accuracy_matrix))
            bwt = float(metric_backward_transfer(accuracy_matrix))
            aaa = float(metric_average_anytime_accuracy(accuracy_matrix))
            return {
                "average_accuracy": average_accuracy,
                "forgetting": forgetting,
                "backward_transfer": bwt,
                "average_anytime_accuracy": aaa,
            }
        except Exception:
            return {
                "average_accuracy": None,
                "forgetting": None,
                "backward_transfer": None,
                "average_anytime_accuracy": None,
            }

    def _write_metrics_summary(self):
        """Write additive machine-readable metrics without affecting CSV logs."""
        if not self.offlog:
            return
        try:
            communication = self.communication_accountant.as_dict()
            server_compute = self.server_compute_timer.as_dict()
            summary = {
                "global": self._summarize_accuracy_matrix(self.global_accuracy_matrix),
                "local": self._summarize_accuracy_matrix(self.local_accuracy_matrix),
                "communication": communication,
                "total_communication_mb": communication["total_mb"],
                "server_compute_seconds": server_compute,
            }
            scheduler = getattr(self, "task_scheduler", None)
            if scheduler is not None:
                summary["async"] = scheduler.metrics()
                if getattr(self.args, "async_mode", False):
                    scheduler.save_schedule(
                        os.path.join(self.save_folder, "async_schedule.json"),
                        run_seed=getattr(self.args, "seed", 0),
                    )
            consolidation = getattr(self, "consolidation_manager", None)
            if consolidation is not None:
                summary["consolidation"] = consolidation.summary()
                consolidation.save_log(
                    os.path.join(self.save_folder, "consolidation_log.json")
                )
            boundary = getattr(self, "boundary_metrics", None)
            if boundary:
                summary.setdefault("robustness", {})["boundary"] = boundary
            scenario = getattr(self, "robustness_scenario", None)
            if scenario is not None and scenario.enabled:
                summary.setdefault("robustness", {})["attacks"] = scenario.summary(
                    getattr(self, "last_client_trust_all", {})
                    or getattr(self, "client_trust", {}),
                    getattr(self, "client_info_dict", {}).keys(),
                )
            if self.heterogeneity_config.enabled:
                summary["heterogeneity"] = self.heterogeneity_metrics or {
                    "enabled": True,
                    "aggregation_mode": self.heterogeneity_config.aggregation_mode,
                    "per_model_type": {},
                    "fairness_gap": None,
                }
                summary["heterogeneity"]["communication_by_architecture"] = (
                    summarize_distillation_communication(
                        self.hetero_communication_records
                    )
                )
                summary["heterogeneity"]["server_distillation_seconds"] = (
                    self.server_distillation_timer.as_dict()
                )
            summary_path = os.path.join(self.save_folder, "metrics_summary.json")
            with open(summary_path, mode="w") as file:
                json.dump(summary, file, indent=2, sort_keys=True, allow_nan=False)
                file.write("\n")
        except Exception:
            pass

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            print(f"Creating client {i} ...")
            if self.args.dataset == 'IMAGENET1k':
                train_data, label_info = read_client_data_FCL_imagenet1k(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
            elif self.args.dataset == 'CIFAR100':
                train_data, label_info = read_client_data_FCL_cifar100(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
            elif self.args.dataset == 'CIFAR10':
                train_data, label_info = read_client_data_FCL_cifar10(i , task=0, classes_per_task=self.args.cpt, count_labels=True)
            elif self.args.dataset == "WILDS":
                train_data, label_info = read_client_data_FCL_wilds(index=i, num_clients=self.num_clients, task=0, count_labels=True)
            else:
                raise NotImplementedError("Not supported dataset")
            print(label_info)
            if self.heterogeneity_config.enabled:
                pool = self.heterogeneity_config.client_model_pool
                offset = int(getattr(self.args, "seed", 0)) % len(pool)
                model_type = pool[(i + offset) % len(pool)]
                client_model = build_client_model(model_type, self.args)
                if self.heterogeneity_config.aggregation_mode == "fedavg":
                    # Homogeneous FedAvg participants start identically.
                    client_model.load_state_dict(self.global_model.state_dict())
                client = clientObj(
                    self.args,
                    id=i,
                    train_data=train_data,
                    model=client_model,
                    model_type=model_type,
                )
            else:
                client = clientObj(self.args, id=i, train_data=train_data)
            self.clients.append(client)

            # update classes so far & current labels
            client.classes_so_far.extend(label_info['labels'])
            client.current_labels.extend(label_info['labels'])
            client.task_dict[0] = label_info['labels']
            client.file_name = self.file_name
            del client, train_data, label_info
            gc.collect()

    def _record_heterogeneity_accuracy(self, per_client_totals):
        if not self.heterogeneity_config.enabled:
            return
        parameter_bytes = {}
        results = []
        for client in self.clients:
            model_type = client.model_type
            parameter_bytes.setdefault(model_type, estimate_model_bytes(client.model))
            totals = per_client_totals.get(client.id, {"correct": 0.0, "samples": 0})
            results.append({"model_type": model_type, **totals})
        metrics = summarize_model_type_accuracy(results, parameter_bytes)
        self.heterogeneity_metrics = {
            "enabled": True,
            "aggregation_mode": self.heterogeneity_config.aggregation_mode,
            **metrics,
        }

    def _record_distillation_communication(self, clients, direction, sample_count):
        """Account float32 classifier logits, grouped by client architecture."""

        payload_bytes = int(sample_count) * int(self.num_classes) * 4
        key = f"{direction}_bytes"
        event_key = f"{direction}_events"
        for client in clients:
            record = self.hetero_communication_records.setdefault(
                client.model_type,
                {
                    "uplink_bytes": 0,
                    "downlink_bytes": 0,
                    "uplink_events": 0,
                    "downlink_events": 0,
                },
            )
            record[key] += payload_bytes
            record[event_key] += 1

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def send_parameters_(self, mode='all', beta=1, selected=False, only_critic=False, gr=False):
        clients = self.clients
        if selected:
            assert (self.selected_clients is not None and len(self.selected_clients) > 0)
            clients = self.selected_clients

        for user in clients:
            if gr == True:
                user.set_parameters_(self.global_generator, beta=beta, only_critic=only_critic, mode=mode, gr=gr,
                                     classifier=self.classifier)  # classifier: from server
            else:
                user.set_parameters_(self.global_generator, beta=beta, only_critic=only_critic, mode=mode, gr=gr)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += len(client.train_data)
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(len(client.train_data))
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def receive_grads(self):

        self.grads = copy.deepcopy(self.uploaded_models)
        # This for copy the list to store all the gradient update value

        for model in self.grads:
            for param in model.parameters():
                param.data.zero_()

        for grad_model, local_model in zip(self.grads, self.uploaded_models):
            for grad_param, local_param, global_param in zip(grad_model.parameters(), local_model.parameters(),
                                                             self.global_model.parameters()):
                grad_param.data = local_param.data - global_param.data
        for w, client_model in zip(self.uploaded_weights, self.grads):
            self.mul_params(w, client_model)

    def mul_params(self, w, client_model):
        for param in client_model.parameters():
            param.data = param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def test_metrics(self, task, glob_iter, flag):
        
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test_metrics(task=task)
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

            test_acc = sum(tot_correct)*1.0 / sum(num_samples)
    
            if flag != "off":
                if flag == "global":
                    subdir = os.path.join(self.save_folder, f"Client_Global/Client_{c.id}")
                    log_key = f"Client_Global/Client_{c.id}/Averaged Test Accurancy"
                elif flag == "local":
                    subdir = os.path.join(self.save_folder, f"Client_Local/Client_{c.id}")
                    log_key = f"Client_Local/Client_{c.id}/Averaged Test Accurancy"

                if self.args.wandb:
                    wandb.log({log_key: test_acc}, step=glob_iter)
                
                if self.offlog:
                    os.makedirs(subdir, exist_ok=True)

                    file_path = os.path.join(subdir, "test_accuracy.csv")
                    file_exists = os.path.isfile(file_path)

                    with open(file_path, mode="w", newline="") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["Step", "Test Accuracy"])  
                        writer.writerow([glob_iter, test_acc]) 

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct

    def test_metrics_(self, task, glob_iter, flag):

        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test_metrics_(task=task)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)

            test_acc = sum(tot_correct) * 1.0 / sum(num_samples)

            if flag != "off":
                if flag == "global":
                    subdir = os.path.join(self.save_folder, f"Client_Global/Client_{c.id}")
                    log_key = f"Client_Global/Client_{c.id}/Averaged Test Accurancy"
                elif flag == "local":
                    subdir = os.path.join(self.save_folder, f"Client_Local/Client_{c.id}")
                    log_key = f"Client_Local/Client_{c.id}/Averaged Test Accurancy"

                if self.args.wandb:
                    wandb.log({log_key: test_acc}, step=glob_iter)

                if self.offlog:
                    os.makedirs(subdir, exist_ok=True)

                    file_path = os.path.join(subdir, "test_accuracy.csv")
                    file_exists = os.path.isfile(file_path)

                    with open(file_path, mode="w", newline="") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["Step", "Test Accuracy"])
                        writer.writerow([glob_iter, test_acc])

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct

    def train_metrics(self, task=None):

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics(task=task)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def eval(self, task, glob_iter, flag):
        stats = self.test_metrics(task, glob_iter, flag=flag)
        stats_train = self.train_metrics(task=task)
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        if flag == "global":
            subdir = os.path.join(self.save_folder, "Global")
            log_keys = {
                "Global/Averaged Train Loss": train_loss,
                "Global/Averaged Test Accuracy": test_acc,
                "Global/Averaged Angle": self.angle_value,
                "Global/Averaged Grads Angle": self.grads_angle_value,
                "Global/Averaged Distance": self.distance_value,
                "Global/Averaged GradNorm": self.norm_value,
            }
            if self.args.tgm and self.selected_clients:
                self.t_angle_after = statistics.mean(client.t_angle_after for client in self.selected_clients)

                log_keys.update({
                    "Global/Timestep Angle After": self.t_angle_after,
                })
                # print(log_keys)

        elif flag == "local":
            subdir = os.path.join(self.save_folder, "Local")
            log_keys = {
                "Local/Averaged Train Loss": train_loss,
                "Local/Averaged Test Accuracy": test_acc,
            }

        if self.args.log and flag == "global":
            # print(f"{sum(stats_train[2])}, {sum(stats_train[1])}")task_id
            print(f"Global Averaged Test Accuracy: {test_acc}")
            print(f"Global Averaged Test Loss: {train_loss}")

        if self.args.log and flag == "local":
            # print(f"{sum(stats_train[2])}, {sum(stats_train[1])}")
            print(f"Local Averaged Test Accuracy: {test_acc}")
            print(f"Local Averaged Test Loss: {train_loss}")

        if self.args.wandb:
            wandb.log(log_keys, step=glob_iter)

        if self.offlog:
            os.makedirs(subdir, exist_ok=True)

            file_path = os.path.join(subdir, "metrics.csv")
            file_exists = os.path.isfile(file_path)

            with open(file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Step", "Train Loss", "Test Accuracy"])  
                writer.writerow([glob_iter, train_loss, test_acc]) 

    # evaluate after end 1 task
    def eval_task(self, task, glob_iter, flag):
        accuracy_on_all_task = []
        per_client_totals = {
            client.id: {"correct": 0.0, "samples": 0} for client in self.clients
        }

        for t in range(self.num_tasks):
            stats = self.test_metrics(task=t, glob_iter=glob_iter, flag="off")
            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
            accuracy_on_all_task.append(test_acc)
            for client_id, samples, correct in zip(*stats):
                per_client_totals[client_id]["correct"] += float(correct)
                per_client_totals[client_id]["samples"] += int(samples)

        self._record_heterogeneity_accuracy(per_client_totals)

        if flag == "global":
            self.global_accuracy_matrix.append(accuracy_on_all_task)
            accuracy_matrix = self.global_accuracy_matrix
            subdir = os.path.join(self.save_folder, "Global")
            log_key = "Global/Averaged Forgetting"
        elif flag == "local":
            self.local_accuracy_matrix.append(accuracy_on_all_task)
            accuracy_matrix = self.local_accuracy_matrix
            subdir = os.path.join(self.save_folder, "Local")
            log_key = "Local/Averaged Forgetting"

        forgetting = metric_average_forgetting(int(task % self.num_tasks), accuracy_matrix)

        if self.args.wandb:
            wandb.log({log_key: forgetting}, step=glob_iter)

        print(f"{log_key}: {forgetting:.4f}")

        if self.offlog:
            os.makedirs(subdir, exist_ok=True)

            csv_filename = os.path.join(subdir, f"{self.args.algorithm}_accuracy_matrix.csv")
            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(accuracy_matrix)

        self._write_metrics_summary()

    # evaluate after end 1 task
    def eval_task_(self, task, glob_iter, flag):
        accuracy_on_all_task = []
        per_client_totals = {
            client.id: {"correct": 0.0, "samples": 0} for client in self.clients
        }

        for t in range(self.num_tasks):
            stats = self.test_metrics_(task=t, glob_iter=glob_iter, flag="off")
            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
            accuracy_on_all_task.append(test_acc)
            for client_id, samples, correct in zip(*stats):
                per_client_totals[client_id]["correct"] += float(correct)
                per_client_totals[client_id]["samples"] += int(samples)

        self._record_heterogeneity_accuracy(per_client_totals)

        if flag == "global":
            self.global_accuracy_matrix.append(accuracy_on_all_task)
            accuracy_matrix = self.global_accuracy_matrix
            subdir = os.path.join(self.save_folder, "Global")
            log_key = "Global/Averaged Forgetting"
        elif flag == "local":
            self.local_accuracy_matrix.append(accuracy_on_all_task)
            accuracy_matrix = self.local_accuracy_matrix
            subdir = os.path.join(self.save_folder, "Local")
            log_key = "Local/Averaged Forgetting"

        forgetting = metric_average_forgetting(int(task % self.num_tasks), accuracy_matrix)

        if self.args.wandb:
            wandb.log({log_key: forgetting}, step=glob_iter)

        print(f"{log_key}: {forgetting:.4f}")

        if self.offlog:
            os.makedirs(subdir, exist_ok=True)

            csv_filename = os.path.join(subdir, f"{self.args.algorithm}_accuracy_matrix.csv")
            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(accuracy_matrix)

        self._write_metrics_summary()

    def assign_unique_tasks(self):
        # Convert lists to sets of tuples for easy comparison
        unique_set = {tuple(task) for task in self.unique_task}
        old_unique_set = {tuple(task) for task in self.old_unique_task}

        # Find new tasks by taking the difference
        new_tasks = unique_set - old_unique_set
        # print(f"new_tasks: {new_tasks}")
        # Loop over new tasks and assign them to task_dict
        for task in new_tasks:
            self.current_task += 1
            self.task_dict[self.current_task] = list(task)

    def cos_sim(self, prev_model, model1, model2):
        prev_param = torch.cat([p.data.view(-1) for p in prev_model.parameters()])
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])

        grad1 = params1 - prev_param
        grad2 = params2 - prev_param

        cos_sim = torch.dot(grad1, grad2) / (torch.norm(grad1) * torch.norm(grad2))
        return cos_sim.item()

    def cosine_similarity(self, model1, model2):
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])
        cos_sim = torch.dot(params1, params2) / (torch.norm(params1) * torch.norm(params2))
        return cos_sim.item()

    def distance(self, model1, model2):
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])

        mse = F.mse_loss(params1, params2)
        return mse.item()

    def spatio_grad_eval(self, model_origin):
        angle = [self.cos_sim(model_origin, self.global_model, models) for models in self.uploaded_models]
        distance = [self.distance(self.global_model, models) for models in self.uploaded_models]
        norm = [self.distance(model_origin, models) for models in self.uploaded_models]
        self.angle_value = statistics.mean(angle)
        self.distance_value = statistics.mean(distance)
        self.norm_value = statistics.mean(norm)
        angle_value = []
        for grad_i in self.grads:
            for grad_j in self.grads:
                angle_value.append(self.cosine_similarity(grad_i, grad_j))
        self.grads_angle_value = statistics.mean(angle_value)
        print(f"grad angle: {self.grads_angle_value}")

    def proto_eval(self, global_model, local_model, task, round):
        # TODO save models to ./pca_eval/file_name/global
        model_filename = f"task_{task}_round_{round}.pth"
        save_dir = os.path.join("pca_eval", self.file_name, "global")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model state_dict
        model_path = os.path.join(save_dir, model_filename)
        torch.save(global_model.state_dict(), model_path)

        save_dir = os.path.join("pca_eval", self.file_name, "local")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model state_dict
        model_path = os.path.join(save_dir, model_filename)
        torch.save(local_model.state_dict(), model_path)
