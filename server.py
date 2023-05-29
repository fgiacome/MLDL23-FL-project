import copy
from collections import OrderedDict
from utils.stream_metrics import StreamSegMetrics
import random
import numpy as np
import torch


class Server:
    def __init__(
        self,
        clients_per_round,
        num_rounds,
        epochs_per_round,
        train_clients,
        test_clients,
        model,
        metrics,
        teacher_update_rounds = None,
        random_state=300890,
    ):
        self.clients_per_round = clients_per_round
        self.num_rounds = num_rounds
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.epochs_per_round = epochs_per_round
        self.teacher_update_rounds = teacher_update_rounds
        # self.metrics = {
        #     'Train': StreamSegMetrics(n_classes = 16, name = 'Mean IoU'),
        #     'Validation': StreamSegMetrics(n_classes = 16, name = 'Mean IoU'),
        #     'Test': StreamSegMetrics(n_classes = 16, name = 'Mean IoU')
        # }
        # The model parameters are saved to a dict and loaded from
        # the same dict when it gets updated
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        # The list of client updates in a round. It is a list of tuples
        # of the form: (training set size, update)
        self.updates = []
        ## This line stays commented for now, plan is
        ## to call random.seed(...) in the notebook explicitly
        ## so as to intuitively restore actually unpredictable behavior
        # self.prng = np.random.default_rng(random_state)
        self.prng = np.random.default_rng()

    def select_clients(self):
        """
        This method selects a random subset of `self.clients_per_round` clients
        from the given traning clients, without replacement.
        :return: list of clients
        """
        num_clients = min(self.clients_per_round, len(self.train_clients))
        return self.prng.choice(self.train_clients, num_clients, replace=False)

    def load_teacher_model_on_clients(self):
        """
        This function loads the centralized model to the clients'
        self training loss object as a teacher.
        """
        for c in self.test_clients + self.train_clients:
            c.update_teacher(self.model_params_dict, strict=False)

    def load_model_on_clients(self):
        """
        This function loads the centralized model to the clients at
        the beginning of each training / testing round.
        """
        for c in self.test_clients + self.train_clients:
            c.model.load_state_dict(self.model_params_dict, strict=False)

    def train_round(self, clients):
        """
        This method trains the model with the dataset of the clients.
        It handles the training at single round level.
        The client updates are saved in the object-level list,
        they will be aggregated.
        :param clients: list of all the clients to train
        """
        train_loss_miou = {str(c): {} for c in clients}

        for i, client in enumerate(clients):
            num_samples = client.get_num_samples()

            # Train the single client model
            loss, miou = client.train(self.epochs_per_round)
            train_loss_miou[str(client)]["Loss"] = loss
            train_loss_miou[str(client)]["mIoU"] = miou

            # Get model parameters
            update = client.generate_update()

            # The list of updates is saved at instance level,
            # but it is also returned as an independent list after each
            # train round.
            self.updates.append((num_samples, update))
        return train_loss_miou

    def aggregate(self):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        # Here we make the average of the updated weights
        total_weight = 0
        base = OrderedDict()
        for client_samples, client_model in self.updates:
            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * value.type(torch.FloatTensor)
        # averaged_sol_n = copy.deepcopy(self.model_params_dict)
        for key, value in base.items():
            if total_weight != 0:
                # averaged_sol_n[key] = value.to('cuda') / total_weight
                self.model_params_dict[key] = value.to("cuda") / total_weight

        # self.model.load_state_dict(averaged_sol_n, strict=False)
        self.model.load_state_dict(self.model_params_dict, strict=False)
        # self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.updates = []

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        :return: list (one elem per round) of dicts (one key per client) of dicts
            (loss, miou) of lists (one elem per epoch) of scalars
        """
        orchestra_statistics = []
        for r in range(self.num_rounds):
            self.load_model_on_clients()
            if self.teacher_update_rounds is not None:
                if self.teacher_update_rounds == 0:
                    if r == 0:
                        self.load_teacher_model_on_clients()
                elif r % self.teacher_update_rounds == 0:
                    self.load_teacher_model_on_clients()
            clients = self.select_clients()
            train_stats = self.train_round(clients)
            orchestra_statistics.append(train_stats)
            self.aggregate()
        return orchestra_statistics

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        :return: dict (one key per client) of dicts (loss, miou) of scalars
        """
        self.load_model_on_clients()
        eval_statistics = {str(c): {} for c in self.train_clients}
        for c in self.train_clients:
            l, m = c.test()
            eval_statistics[str(c)]["Loss"] = l
            eval_statistics[str(c)]["mIoU"] = m
        return eval_statistics

    def test(self):
        """
        This method handles the test on the test clients
        :return: dict (one key per client) of dicts (loss, miou) of scalars
        """
        self.load_model_on_clients()
        eval_statistics = {str(c): {} for c in self.test_clients}
        for c in self.test_clients:
            l, m = c.test()
            eval_statistics[str(c)]["Loss"] = l
            eval_statistics[str(c)]["mIoU"] = m
        return eval_statistics
