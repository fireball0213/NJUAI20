import numpy as np
from arguments import get_args
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.transforms import transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_args()
class DaggerAgent:
    def __init__(self, ):
        pass
    @abstractmethod
    def select_action(self, ob):
        pass

# here is an example of creating your own Agent
class ExampleAgent(DaggerAgent):
    def __init__(self, necessary_parameters=None):
        super(DaggerAgent, self).__init__()
        # init your model
        self.model = None
    # train your model with labeled data
    def update(self, data_batch, label_batch):
        self.model.train(data_batch, label_batch)
    # select actions by your model
    def select_action(self, data_batch):
        label_predict = self.model.predict(data_batch)
        return label_predict

class MyAgent(DaggerAgent):
    def __init__(self, necessary_parameters=None):
        super(DaggerAgent, self).__init__()
        # init your model
        self.batch_size = args.num_steps
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.epochs = 10
        self.num_classes = 8
        self.learning_rate = 0.001
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = resnet18(num_classes=self.num_classes).to(device)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    # train your model with labeled data
    def update(self, data_batch, label_batch):
        data_batch = np.array(data_batch)
        label_batch = np.array(label_batch)
        """
        # shuffle
        indices = np.arange(data_batch.shape[0])
        np.random.shuffle(indices)
        data_batch = data_batch[indices]
        label_batch = label_batch[indices]
        """
        label_batch = torch.from_numpy(label_batch)
        label_batch = label_batch.to(device)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, len(data_batch), self.batch_size):
                inputs = torch.stack([self.transform(img) for img in data_batch[i:i + self.batch_size]])
                inputs = inputs.to(device)
                labels = label_batch[i:i + self.batch_size]
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print("Epoch: {}, Loss: {}.".format(epoch + 1, total_loss))

    # select actions by your model
    def select_action(self, data_batch):
        self.model.eval()
        inputs = self.transform(data_batch)
        inputs = torch.unsqueeze(inputs, dim=0)
        inputs = inputs.to(device)
        label_predict = self.model(inputs)
        label_predict = torch.squeeze(label_predict, dim=0)
        label_predict = label_predict.cpu().detach().numpy()
        label_predict = np.argmax(label_predict, axis=0)
        # print(label_predict)
        return label_predict
