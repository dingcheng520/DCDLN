import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
from classic_models.alexnet import AlexNet
from classic_models.alexnet import Lenet, Lenet_1
from classic_models.googlenet_v1 import GoogLeNet
from classic_models.vision_transformer import vit_base_patch16_224
from classic_models.resnet import resnet34
from classic_models.mobilenet_v2 import mobilenet_v2
from classic_models.densenet import densenet121
import time
from classic_models.vggnet import vgg19, vgg11
from classic_models.shufflenet_v2 import shufflenet_v2_x1_0
from classic_models.zfnet import zfnet
from torchsummary import summary
import pandas as pd

def main():
    # 判断可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 注意改成自己的数据集路径
    data_path = "F:\\研究\\paper1\\code\\feature_extraction\\data"

    assert os.path.exists(data_path), "{} path does not exist.".format(data_path)

    # 数据预处理与增强
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    # 加载数据集
    # dataset = datasets.ImageFolder(root=data_path, transform=data_transform["train"])
    dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=data_transform["train"])
    test_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=data_transform["val"])
    num_samples = len(dataset)
    num_test = len(test_dataset)
    print("using {} images for training and validation.".format(num_samples))
    print("using {} images for test.".format(num_test))

    batch_size = 8

    # 10折交叉验证
    k_folds = 10
    fold_size = num_samples // k_folds

    # results_df = pd.DataFrame(columns=['Fold', 'Epoch', 'Val Loss', 'Val Acc', 'Test Loss', 'Test Acc'])
    results_list = []
    for fold in range(k_folds):
        # 划分训练集和验证集
        print("using {} fold for trainging and validation".format(fold))
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size

        val_indices = list(range(val_start, val_end))
        train_indices = list(range(0, val_start)) + list(range(val_end, num_samples))
        print("val_indices is {}.".format(val_indices))
        print("The length of (val_indices) is {}, the length of (train_indices) is {}".format(len(val_indices), len(train_indices)))

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        # 使用subset sampler创建data loader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 创建模型
        # model = densenet121(num_classes=2)
        # model = AlexNet(num_classes=2)  # 分2类
        # model = resnet34(num_classes=2)
        # model = mobilenet_v2(num_classes=2)
        model = shufflenet_v2_x1_0(num_classes=2)
        model = nn.DataParallel(model)
        model.to(device)

        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # 定义优化器
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        # 训练参数
        num_epochs = 10

        save_path = os.path.abspath(os.path.join(os.getcwd(), './k_fold_result/weights/shufflenet_v2'))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 训练模型
        best_acc = 0.0
        best_model_weights = None

        for epoch in range(num_epochs):
            model.train()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            train_loss = running_loss / len(train_indices)
            train_acc = running_corrects.double() / len(train_indices)

            model.eval()

            val_loss = 0.0
            val_corrects = 0

            test_loss = 0.0
            test_corrects = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)

                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item() * inputs.size(0)
                    test_corrects += torch.sum(preds == labels.data)

            val_loss = val_loss / len(val_indices)
            val_acc = val_corrects.double() / len(val_indices)

            test_loss = test_loss / num_test
            test_acc = test_corrects.double() / num_test

            # results_df = results_df.append({
            #     'Fold': fold,
            #     'Epoch': epoch,
            #     'Val Loss': val_loss,
            #     'Val Acc': val_acc,
            #     'Test Loss': test_loss,
            #     'Test Acc': test_acc
            # }, ignore_index=True)

            results_list.append({
                'Fold': fold,
                'Epoch': epoch,
                'Train acc':train_acc,
                # 'Val Loss': val_loss,
                'Val Acc': val_acc,
                # 'Test Loss': test_loss,
                'Test Acc': test_acc,
                'Train Loss':train_loss,
                'Val Loss': val_loss,
                'Test Loss': test_loss
            })

            results_df = pd.DataFrame(results_list)
            results_df.to_excel('F:\\研究\\paper1\\code\\feature_extraction\\k_fold_result\\weights\\shufflenet_v2\\results.xlsx', index=False)


            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Validation Loss: {:.4f}, Validation Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))

            print('Epoch [{}/{}], Test Loss: {:.4f}, Test Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, test_loss, test_acc))

            if val_acc >= best_acc:
                best_epoch = epoch
                best_acc = val_acc
                best_model_weights = model.state_dict()

        # 在每个fold结束后，保存最佳模型
        best_model_file = f"best_model_fold_{fold}.pth"
        # torch.save(best_model_weights, best_model_file)
        # torch.save(best_model_weights, os.path.join(save_path, "densenet121.pth"))
        torch.save(best_model_weights, os.path.join(save_path, "best_model_fold_{}.pth".format(fold)))
        print("the best epoch is {}".format(best_epoch))
        print("Saved best model for fold {}.".format(fold))

if __name__ == '__main__':
    main()