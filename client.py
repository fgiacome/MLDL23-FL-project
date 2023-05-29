import copy
import torch
from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.stream_metrics import StreamSegMetrics
from utils.utils import HardNegativeMining, MeanReduction, SelfTrainingLoss

class NoTeacherException(Exception):
    pass

class Client:
    def __init__(
        self,
        client_dataset,
        batch_size,
        model,
        dataloader="train",
        optimizer="Adam",
        lr=1e-3,
        device="cpu",
        reduction="MeanReduction",
        self_training = False
    ):
        # Client dataset and attributes
        self.dataset = client_dataset
        self.name = client_dataset.client_name
        self.model = model
        self.device = device
        self.learning_rate = lr
        self.self_training = self_training
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
        self.reduction = (
            MeanReduction() if reduction == "MeanReduction" else HardNegativeMining()
        )

        if self.self_training is True:
            self.criterion = SelfTrainingLoss()

        # DataLoader initialization
        if dataloader == "train":
            self.dataloader = DataLoader(
                client_dataset, batch_size=batch_size, shuffle=True, drop_last=True
            )
        elif dataloader == "test":
            self.dataloader = DataLoader(client_dataset, batch_size=8, shuffle=False)
        else:
            raise NotImplementedError

        # Number of samples on which the client train = n_batches * batch_size
        self.num_samples = len(self.dataloader) * batch_size

        # Optimizer initialization
        if optimizer == "Adam" or optimizer == "SGD":
            self.optimizer = optimizer
        else:
            raise NotImplementedError

    def __str__(self):
        return self.name

    def run_epoch(self, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        # Mean_IoU initialization
        mean_iou = StreamSegMetrics(n_classes=16, name="Mean IoU")

        # Cumulative loss initialization
        cumulative_loss = 0

        # Iterations over batches
        for images, labels in self.dataloader:
            # Send data to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Reset gradient
            optimizer.zero_grad()

            # Forward
            labels_hat = self.model(images)["out"]
            labels_pred = torch.argmax(labels_hat, dim=1)

            # Compute loss
            if self.self_training is False:
                loss = self.criterion(labels_hat, labels)
                loss = self.reduction(loss, labels)
            else:
                loss = self.criterion.forward(labels_pred, images)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update loss and metrics
            cumulative_loss += loss.item()
            mean_iou.update(labels.cpu().numpy(), labels_pred.cpu().numpy())

        return cumulative_loss / self.num_samples, mean_iou.get_results()

    def train(self, num_epochs=1):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """

        # Optimizer initialization
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )

        # Model's train mode
        self.model.train()

        # Lists initializations
        loss_list = [0] * num_epochs
        mean_iou_list = [0] * num_epochs

        # Running over epochs
        for epoch in range(num_epochs):
            # Run epoch
            loss, mean_iou = self.run_epoch(optimizer)

            # Update lists
            loss_list[epoch] = loss
            mean_iou_list[epoch] = mean_iou

        return loss_list, mean_iou_list

    def test(self):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """

        # This method tests the model on local images
        # Model's evaluation mode
        self.model.eval()

        # Metric initialization
        mean_iou = StreamSegMetrics(n_classes=16, name="Mean IoU")

        # Cumulative loss initialization
        cumulative_loss = 0

        # Iterations over batches
        with torch.no_grad():
            for images, labels in self.dataloader:
                # Send images and labels to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Compute predictions
                labels_hat = self.model(images)["out"]
                labels_pred = torch.argmax(labels_hat, dim=1)

                # Compute loss
                loss = self.criterion(labels_hat, labels)
                loss = self.reduction(loss, labels)

                # Update loss and metrics
                cumulative_loss += loss.item()
                mean_iou.update(labels.cpu().numpy(), labels_pred.cpu().numpy())

        return cumulative_loss / self.num_samples, mean_iou.get_results()

    def generate_update(self):
        return copy.deepcopy(self.model.state_dict())

    def get_num_samples(self):
        return self.num_samples

    def set_teacher(self, model):
        if isinstance(self.self_training, SelfTrainingLoss):
            raise NoTeacherException
        self.criterion.set_teacher(model)
    
    def update_teacher(self, state_dict):
        if isinstance(self.self_training, SelfTrainingLoss):
            raise NoTeacherException
        self.criterion.teacher.load_state_dict(state_dict, strict=False)