import copy
from collections import OrderedDict
from utils.stream_metrics import StreamSegMetrics
import random
import numpy as np
import torch
import client


class Server:
    def __init__(
        self,
        clients_per_round,
        num_rounds,
        epochs_per_round,
        train_clients: list[client.Client],
        test_clients: list[client.Client],
        model: torch.nn.Module,
        use_prior,
        n_rounds_no_prior,
        random_state=300890,
    ):
        self.clients_per_round = clients_per_round
        self.num_rounds = num_rounds
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.epochs_per_round = epochs_per_round
        self.use_prior = use_prior
        self.n_rounds_no_prior = (
            int(n_rounds_no_prior * num_rounds) if self.use_prior == False else 1
        )
        self.weights = OrderedDict(
            (client.client_id, 1 / len(self.train_clients))
            for client in self.train_clients
        )
        self.weights_track = [self.weights.copy()]
        self.epochs_stds = None
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
        self.prng = np.random.default_rng(random_state)

    def select_clients(self):
        """
        This method selects a random subset of `self.clients_per_round` clients
        from the given traning clients, without replacement.
        :return: list of clients
        """
        if self.use_prior == True:
            return self.prng.choice(
                self.train_clients,
                size=self.num_clients_per_round,
                replace=False,
                p=[w for _, w in self.weights.items()],
            )
        else:
            return self.prng.choice(
                self.train_clients, size=self.clients_per_round, replace=False
            )

    def load_model_on_clients(self):
        """
        This function loads the centralized model to the clients at
        the beginning of each training / testing round.
        """
        for c in self.test_clients + self.train_clients:
            c.model.load_state_dict(self.model_params_dict, strict=False)

    def train_round(self, clients: list[client.Client]):
        """
        This method trains the model with the dataset of the clients.
        It handles the training at single round level.
        The client updates are saved in the object-level list,
        they will be aggregated.
        :param clients: list of all the clients to train
        """
        # Train function in client should return the loss series
        # and the mIoU series
        client_loss = OrderedDict()
        for client in clients:
            # Train the client
            client_loss[client.client_id] = client.train(self.epochs_per_round)[0]

            # Weigths are used to weight the client update, rather than the
            # number of training samples.
            self.updates.append((client.client_id, client.generate_update()))
        return client_loss

    def update_weights(self, client_loss: OrderedDict):
        """
        Updates the weigths saved at instance level.
        :param client_loss: a dictionary client_id -> list of round losses
        :return: the sum of the weights of the selected clients
        """
        # Normalize selected clients' weights
        w_sum = 0
        for client_id in client_loss.keys():
            w_sum += self.weights[client_id]
        for client_id in client_loss.keys():
            self.weights[client_id] = self.weights[client_id] / w_sum

        # Compute the mean process
        weights_sum = 0
        weights2_sum = 0
        loss_tensor = torch.empty(len(client_loss), self.epochs_per_round)
        mean_loss = torch.zeros(self.epochs_per_round)
        for idx, client_id in enumerate(client_loss.keys()):
            loss_tensor[idx] = client_loss[client_id]
            mean_loss += self.weights[client_id] * client_loss[client_id]
            weights_sum += self.weights[client_id] / len(client_loss)
            weights2_sum += (self.weights[client_id] ** 2) / len(client_loss)

        # Compute the standard deviation for each epoch
        sigma2 = ((loss_tensor - mean_loss) ** 2).sum(dim=0) / (len(client_loss) - 1)
        std_loss = sigma2 * (weights2_sum / weights_sum)

        self.epochs_stds = torch.sqrt(std_loss)

        # Compute the rewards for each epoch and each client
        exp_args_tensor = 0.5 * ((loss_tensor - mean_loss) ** 2) / std_loss
        reward_tensor = torch.exp(-exp_args_tensor).mean(dim=1)
        reward_tensor = reward_tensor / reward_tensor.sum()

        # Assign rewards to weights
        for idx, client_id in enumerate(client_loss.keys()):
            self.weights[client_id] = reward_tensor[idx].item() * w_sum
        return w_sum

    def aggregate(self, inv_scale_factor):
        """
        This method handles the FedAvg aggregation
        :param inv_scale_factor: scales the weights by the inverse of this factor
        :return: aggregated parameters
        """
        # Here we make the average of the updated weights
        base = OrderedDict()
        for client_id, client_model in self.updates:
            for key, value in client_model.items():
                if key in base:
                    base[key] += (
                        (1 / inv_scale_factor)
                        * self.weights[client_id]
                        * value.type(torch.FloatTensor)
                    )
                else:
                    base[key] = (
                        (1 / inv_scale_factor)
                        * self.weights[client_id]
                        * value.type(torch.FloatTensor)
                    )
        # averaged_sol_n = copy.deepcopy(self.model_params_dict)
        for key, value in base.items():
            # averaged_sol_n[key] = value.to('cuda') / total_weight
            self.model_params_dict[key] = value.to("cuda")

        # self.model.load_state_dict(averaged_sol_n, strict=False)
        self.model.load_state_dict(self.model_params_dict, strict=False)
        # self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.updates = []

    def train(self, path=None):
        """
        This method orchestrates the training the evals and tests at rounds level
        :return: Train / test statistics at each round. "Train as it happens" is the typical epoch loss returned from each client.
        """
        orchestra_statistics = {"Train": [], "Test": [], "Train as it happens": []}

        for r in range(self.num_rounds):
            self.load_model_on_clients()
            if (r >= self.n_rounds_no_prior) and (self.use_prior == False):
                self.use_prior = True
            print(f"Round {r + 1}")
            clients = self.select_clients()
            clients_loss = self.train_round(clients)
            orchestra_statistics["Train as it happens"].append(clients_loss)
            constant_w_sum = self.update_weights(clients_loss)
            self.aggregate(constant_w_sum)

            # normalize all weights (this normalization pass is extra)
            weights_sum = 0
            for _, w in self.weights.items():
                weights_sum += w
            for client_id, w in self.weights.items():
                self.weights[client_id] = w / weights_sum
            self.weights_track.append(self.weigths.copy())

            # compute mean accuracy on train set
            acc = 0
            stats = self.eval_train()
            for _, res in stats.items():
                acc += res["Accuracy"] / len(self.train_clients)
            orchestra_statistics["Train"].append(acc)

            # compute mean accuracy on test set
            acc = 0
            stats = self.test()
            for _, res in stats.items():
                acc += res["Accuracy"] / len(self.test_clients)
            orchestra_statistics["Test"].append(acc)
            
            if path is not None:
                self.save_checkpoint(path + f"_{r}.json", r)
        return orchestra_statistics

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        :return: dict (one key per client) of dicts (loss, miou) of scalars
        """
        self.load_model_on_clients()
        eval_statistics = {c.client_id: {} for c in self.train_clients}
        for c in self.train_clients:
            l, m = c.test()
            eval_statistics[c.client_id]["Loss"] = l
            eval_statistics[str(c)]["mIoU"] = m
        return eval_statistics

    def test(self):
        """
        This method handles the test on the test clients
        :return: dict (one key per client) of dicts (loss, miou) of scalars
        """
        self.load_model_on_clients()
        eval_statistics = {c.client_id: {} for c in self.test_clients}
        for c in self.test_clients:
            l, m = c.test()
            eval_statistics[c.client_id]["Loss"] = l
            eval_statistics[c.client_id]["mIoU"] = m
        return eval_statistics
    
    def save_checkpoint(self, path, round):
        torch.save(
            {
                "round": round,
                "model_state_dict": self.model.state_dict()
            }, path
        )
