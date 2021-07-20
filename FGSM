from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
torch.manual_seed(42)
# training and testing functions
class MNISTResNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)

def train(model, train_loader, valid_loader, lr=.01, momentum=.9, epochs=1000, weight_decay=1e-6, device="cuda"):
    """Train a classifier."""
    model.to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=.1)
    # train_loss_vals = []
    # valid_loss_vals = []
    # valid_acc_vals = []
    for epoch in range(epochs):
        train_loss = 0.
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        # # train_loss_vals.append(train_loss)

        with torch.no_grad():
            model.eval()
            valid_loss = 0.
            correct = 0.
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                y_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
                valid_loss += loss.item()
                # valid_acc += torch.mean(torch.eq(y, y_pred).float()).item()
                correct += torch.sum(y == y_pred).item()
            valid_loss /= len(valid_loader)
            valid_acc = correct / len(valid_loader.dataset)
            # valid_loss_vals.append(valid_loss)
            # valid_acc_vals.append(valid_acc)
        model.train()

        if epoch % 10 == 0:
            print(f"[epoch {epoch}] train loss: {train_loss}, valid loss: {valid_loss}, valid acc: {valid_acc}")

    return model

def test(model, test_loader, adversarial=False, device="cuda"):
    """Test a classifier, optionally applying adversarial
       perturbations to inputs.
    """
    model.to(device).eval()
    # test_acc = 0.
    correct = 0.
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with torch.enable_grad():
                    x = fgsm(x, y, model)
            logits = model(x)
            y_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
            correct += torch.sum(y == y_pred).item()

    test_acc = correct / len(test_loader.dataset)
    return test_acc

# adversarial attacks
def fgsm(x, y, model, eps=.25):
    """Fast gradient sign method for adversarial attack."""
    x.requires_grad = True
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    return x + eps * torch.sign(x.grad)

def main(args):
    device = "cpu"# if args.no_gpu else "cuda"
    model = MNISTResNet()
    transform = Compose([ToTensor(),Normalize((0.1300,), (0.3071,))])
    train_data = MNIST("./data", train=True, transform=transform, download=True)
    print(len(train_data))
    train_data = Subset(train_data, range(0, len(train_data) - 5000))
    print(len(train_data))
    valid_data = Subset(train_data, range(len(train_data) - 5000, len(train_data)))
    print(len(valid_data))
    test_data = MNIST("./data",transform=transform, train=False)

    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data,batch_size=args.batch_size,num_workers=args.num_workers)
    test_loader = DataLoader(test_data,batch_size=args.batch_size,num_workers=args.num_workers)

    cross_entropy_model = train(model,train_loader=train_loader,valid_loader=valid_loader,lr=args.lr,
                                momentum=args.momentum,
                                epochs=args.epochs,
                                weight_decay=args.weight_decay,
                                device=device)

    test_acc = test(model, test_loader, adversarial=False, device=device)
    test_acc_adversarial = test(model, test_loader, adversarial=True, device=device)
    print(f"test accuracy: {test_acc}")
    print(f"adversarial accuracy: {test_acc_adversarial}")

if __name__  == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr",type=float,default=.01,help="The learning rate.")
    parser.add_argument("--momentum",type=float,default=.9,help="The momentum value.")
    parser.add_argument("--epochs",type=int,default=1000,help="Number of training epochs.")
    parser.add_argument("--batch_size",type=int,default=64,help="The batch size.")
    parser.add_argument("--weight_decay",type=float,default=1e-6,help="The weight decay value.")
    parser.add_argument("--num_workers",type=int,default=1,help="Number of worker processes to use when loading data.")
    parser.add_argument("--no_gpu",action="store_true",help="Disable GPU training.")
    args = parser.parse_args()
    main(args)
