import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction


class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

    def __str__(self):
        return self.name

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        
    def run_epoch(self, cur_epoch, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        # Count the number of samples, it should euqal the size of the client dataset
        samples = 0
        epoch_loss = float(0)
        for cur_step, (images, labels) in enumerate(self.train_loader):
            # Send data to GPU
            images = images.cuda()
            labels = labels.cuda()

            # Reset the gradients
            optimizer.zero_grad()

            # Predictions
            labels_hat = self._get_outputs(images)

            # Compute **unreduced** loss
            loss = self.criterion(labels_hat, labels)

            # MeanReduction computes the mean of the loss of each pixel;
            # HardNegativeMining computes the mean of the top-25% pixel-losses;
            # In both cases, the pixels are pooled together (there is no
            # distinction for pixels of different images).
            loss = self.reduction(loss)
            epoch_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
        return samples, epoch_loss / samples

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-3)
        self.model.train()
        for epoch in range(self.args.num_epochs):
            # TODO: save history of epoch losses, to print / debug later
            self.run_epoch(epoch, optimizer)
        return len(self.dataset)

    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                labels_hat = self._get_outputs(images)
                self.update_metric(metric, labels_hat, labels)

    def generate_update(self):
        return copy.deepcopy(self.model.state_dict())
    
    def num_samples(self):
        return len(self.dataset)