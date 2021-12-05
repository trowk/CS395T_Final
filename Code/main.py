
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import time
from perceiver_pytorch import Perceiver
import torchvision.models as models

def DataLoaders(dataset = "CF-10", data_root = "./", batch_size = 16):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ColorJitter(brightness=0.5, hue = 0.25),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    root_dir = os.path.join(data_root, dataset)
    if dataset == "CF-10":
        train_dataset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    elif dataset == "STL-10":
        train_dataset = datasets.STL10(root=root_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.STL10(root=root_dir, train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def GetNumberParameters(model):
  return sum(np.prod(p.shape).item() for p in model.parameters())

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


from vit_pytorch import ViT
import torchvision
from vit_pytorch import ViT


class ResidualCNNBlock(torch.nn.Module):
    def __init__(self, c_in, c_out, should_stride=False):
        super().__init__()

        if should_stride:
            stride = 2
        else:
            stride = 1

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_out, 3, padding=1, stride=stride),
            torch.nn.BatchNorm2d(c_out),
        )

        self.relu = torch.nn.ReLU()

        if c_in != c_out or should_stride:
            self.identity = torch.nn.Conv2d(c_in, c_out, 1, stride=stride)
        else:
            self.identity = lambda x: x

    def forward(self, x):
        result = self.block(x)
        x = self.identity(x)

        return self.relu(x + result)


class ResMMBlock(torch.nn.Module):
    def __init__(self, image_size=32, num_labels=10, depth=1, att_heads=1, mlp_dim=2048, output_dim=1024):
        super().__init__()
        self.conv_net = torch.nn.Sequential(
            ResidualCNNBlock(3, 32, False),
            ResidualCNNBlock(32, 32, False),
            ResidualCNNBlock(32, 128, False),
            ResidualCNNBlock(128, 128, False),
            ResidualCNNBlock(128, 3, False)
        )
        self.vision_transformer = ViT(
            image_size=image_size,
            patch_size=image_size // 16,
            num_classes=num_labels,
            dim=output_dim,
            depth=depth,
            heads=att_heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.vision_transformer = torch.nn.Sequential(*(list(self.vision_transformer.children())[:-1]))

        self.upsample_vit = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(3))

    def forward(self, x):
        res = self.conv_net(x)
        res = self.vision_transformer(res)
        res = res[:, :, :, None]
        dim = int(res.shape[2] ** 0.5)
        res = res.view(res.shape[0], res.shape[1], dim, dim)

        return self.upsample_vit(res)


class ResAlternatingMixtureModel(torch.nn.Module):
    def __init__(self, num_blocks=1, image_size=32, num_labels=10, depth=1, att_heads=1, mlp_dim=2048, output_dim=1024):
        super().__init__()
        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(
                ResMMBlock(image_size=image_size, num_labels=num_labels, depth=depth, att_heads=att_heads,
                           mlp_dim=mlp_dim, output_dim=output_dim))
        self.ffn = torch.nn.Linear(3 * 32 * 32, 10)
        self.MM = torch.nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.ffn(self.MM(x).flatten(start_dim=1))


# class JointMixtureModel(torch.nn.Module):
#     def __init__(self, num_blocks=1, image_size=32, num_labels=10, depth=2, att_heads=2, mlp_dim=2048, output_dim=1024):
#         super().__init__()
#         self.blocks = []
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         conv_nets = list()
#         for i in range(num_blocks):
#             conv_nets.append(torch.nn.Sequential(
#                 ResidualCNNBlock(3, 32, False),
#                 ResidualCNNBlock(32, 32, False),
#                 ResidualCNNBlock(32, 128, False),
#                 ResidualCNNBlock(128, 128, False),
#                 ResidualCNNBlock(128, 3, False),
#             ))
#         self.conv_net = torch.nn.Sequential(*conv_nets)
#             vit = ViT(
#                 image_size=image_size,
#                 patch_size=image_size // 16,
#                 num_classes=num_labels,
#                 dim=output_dim,
#                 depth=depth,
#                 heads=att_heads,
#                 mlp_dim=mlp_dim,
#                 dropout=0.1,
#                 emb_dropout=0.1
#             )
#             vit = torch.nn.Sequential(*(list(vit.children())[:-1]))
#
#             upsample_vit = torch.nn.Sequential(
#                 torch.nn.ConvTranspose2d(256, 3, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(3))
#             conv_net = conv_net.to(device)
#             vit = vit.to(device)
#             upsample_vit = upsample_vit.to(device)
#             self.blocks.append([conv_net, vit, upsample_vit])
#
#         self.ffn = torch.nn.Linear(3 * 32 * 32, 10)
#
#     def forward(self, x):
#         input = x
#         for conv_net, vit, upsample in self.blocks:
#             res1 = conv_net(input)
#             res2 = vit(input)
#             res2 = res2[:, :, :, None]
#             dim = int(res2.shape[2] ** 0.5)
#             res2 = res2.view(res2.shape[0], res2.shape[1], dim, dim)
#             res2 = upsample(res2)
#
#             input = res1 + res2
#
#         return self.ffn(input.flatten(start_dim=1))

def JointMixtureModel(num_blocks = 1):
    if num_blocks == 1:
        return JointMixtureModel1()
    else:
        return JointMixtureModel2()

class JointMixtureModel1(torch.nn.Module):
    def __init__(self, num_blocks=1, image_size=32, num_labels=10, depth=2, att_heads=2, mlp_dim=2048, output_dim=1024):
        super().__init__()
        self.conv_net = torch.nn.Sequential(
            ResidualCNNBlock(3, 32, False),
            ResidualCNNBlock(32, 32, False),
            ResidualCNNBlock(32, 128, False),
            ResidualCNNBlock(128, 128, False),
            ResidualCNNBlock(128, 3, False),
        )
        self.vit = ViT(
            image_size=image_size,
            patch_size=image_size // 16,
            num_classes=num_labels,
            dim=output_dim,
            depth=depth,
            heads=att_heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.vit = torch.nn.Sequential(*(list(self.vit.children())[:-1]))

        self.upsample_vit = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(3))

        self.ffn = torch.nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        res1 = self.conv_net(x)
        res2 = self.vit(x)
        res2 = res2[:, :, :, None]
        dim = int(res2.shape[2] ** 0.5)
        res2 = res2.view(res2.shape[0], res2.shape[1], dim, dim)
        res2 = self.upsample_vit(res2)

        input = res1 + res2

        return self.ffn(input.flatten(start_dim=1))

class JointMixtureModel2(torch.nn.Module):
    def __init__(self, num_blocks=1, image_size=32, num_labels=10, depth=2, att_heads=2, mlp_dim=2048, output_dim=1024):
        super().__init__()
        self.conv_net1 = torch.nn.Sequential(
            ResidualCNNBlock(3, 32, False),
            ResidualCNNBlock(32, 32, False),
            ResidualCNNBlock(32, 128, False),
            ResidualCNNBlock(128, 128, False),
            ResidualCNNBlock(128, 3, False),
        )

        self.conv_net2 = torch.nn.Sequential(
            ResidualCNNBlock(3, 32, False),
            ResidualCNNBlock(32, 32, False),
            ResidualCNNBlock(32, 128, False),
            ResidualCNNBlock(128, 128, False),
            ResidualCNNBlock(128, 3, False),
        )
        self.vit1 = ViT(
            image_size=image_size,
            patch_size=image_size // 16,
            num_classes=num_labels,
            dim=output_dim,
            depth=depth,
            heads=att_heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.vit1 = torch.nn.Sequential(*(list(self.vit1.children())[:-1]))

        self.upsample_vit1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(3))

        self.vit2 = ViT(
            image_size=image_size,
            patch_size=image_size // 16,
            num_classes=num_labels,
            dim=output_dim,
            depth=depth,
            heads=att_heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.vit2 = torch.nn.Sequential(*(list(self.vit2.children())[:-1]))

        self.upsample_vit2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(3))

        self.ffn = torch.nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        res1 = self.conv_net1(x)
        res2 = self.vit1(x)
        res2 = res2[:, :, :, None]
        dim = int(res2.shape[2] ** 0.5)
        res2 = res2.view(res2.shape[0], res2.shape[1], dim, dim)
        res2 = self.upsample_vit1(res2)

        x = res1 + res2

        res1 = self.conv_net2(x)
        res2 = self.vit2(x)
        res2 = res2[:, :, :, None]
        dim = int(res2.shape[2] ** 0.5)
        res2 = res2.view(res2.shape[0], res2.shape[1], dim, dim)
        res2 = self.upsample_vit2(res2)

        x = res1 + res2

        return self.ffn(x.flatten(start_dim=1))


def train(num_blocks=1, model_type="alternating", batch_size = 64, dataset = "CF-10", epochs=50):
    result = list()
    lr = 1e-3
    num_workers = 2
    weight_decay = 1e-4
    # Set up the cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == "joint":
        model = JointMixtureModel(num_blocks=num_blocks).to(device)
    elif model_type == "alternating":
        model = ResAlternatingMixtureModel(num_blocks=num_blocks).to(device)
    elif model_type == "resnet18":
        model = models.resnet18(pretrained=False).to(device)
    elif model_type == "VIT":
        model = ViT(
                image_size=32,
                patch_size=32 // 16,
                num_classes=10,
                dim=1024,
                depth=2,
                heads=4,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            ).to(device)

    num_params = GetNumberParameters(model)

    # Set up loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Set up training data and validation data
    data_train, data_val = DataLoaders(dataset, batch_size=batch_size)

    # Set up loggers
    # log_dir = '.'
    # log_time = '{}'.format(time.strftime('%H-%M-%S'))
    # log_name = 'lr=%s_epoch=%s_batch_size=%s_wd=%s' % (lr, epochs, batch_size, weight_decay)
    # logger = tb.SummaryWriter()
    # train_logger = tb.SummaryWriter(path.join(log_dir, 'train') + '/%s_%s' % (log_name, log_time))
    # valid_logger = tb.SummaryWriter(path.join(log_dir, 'test') + '/%s_%s' % (log_name, log_time))
    # global_step = 0
    # Wrap in a progress bar.

    for epoch in range(epochs):
        print("!!!!!!!!!!!!!!EPOCH {}!!!!!!!!!!!!!".format(epoch))
        # Set the model to training mode.
        model.train()

        train_accuracy_val = list()
        s = time.time()
        for x, y in tqdm(data_train):
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            train_accuracy_val.append(accuracy(y_pred, y))

            # Compute loss and update model weights.
            loss = loss_func(y_pred, y)

            loss.backward()
            optim.step()
            optim.zero_grad()

            # Add loss to TensorBoard.
            # train_logger.add_scalar('Loss', loss.item(), global_step=global_step)
            # global_step += 1

        train_accuracy_total = torch.FloatTensor(train_accuracy_val).mean().item()
        train_time = time.time() - s
        # train_logger.add_scalar('Train Accuracy', train_accuracy_total, global_step=global_step)
        # print("Train Accuracy: {:.4f}".format(train_accuracy_total))

        # Set the model to eval mode and compute accuracy.
        # No need to change this, but feel free to implement additional logging.
        model.eval()

        accuracys_val = list()
        s = time.time()
        for x, y in data_val:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            accuracys_val.append(accuracy(y_pred, y))
        val_time = time.time() - s

        accuracy_total = torch.FloatTensor(accuracys_val).mean().item()
        result.append({"epoch": epoch,
                       "batch_size": batch_size,
                       "model_type": model_type,
                       "model_parameters": num_params,
                       "train_accuracy": train_accuracy_total,
                       "validation_accuracy": accuracy_total,
                       "train_time": train_time,
                       "val_time": val_time})
        # print("Validation Accuracy: {:.4f}".format(accuracy_total))
        # valid_logger.add_scalar('Validation Accuracy', accuracy_total, global_step=global_step)
    # print(time.time() - s)
    return result

if __name__ == "__main__":
    import csv
    with open('C:\\Users\\trowb\\OneDrive\\Documents\\results.csv','w') as f:
        w = csv.DictWriter(f, ["epoch", "batch_size", "model_type", "model_parameters", "train_accuracy", "validation_accuracy", "train_time", "val_time"])
        for dataset in ["CF-10", "STL-10"]:
            for model, blocks, batch_size in [("joint", 1, 64), ("joint", 2, 32), ("alternating", 1, 64), ("alternating", 2, 32), ("resnet18", 1, 64), ("VIT", 1, 64)]:
                results = train(num_blocks=blocks, model_type=model, batch_size=batch_size, dataset=dataset, epochs=50)
                w.writerows(results)
