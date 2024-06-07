import os
import time
import sys
import json
import json5

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from MedMamba import VSSM as medmamba  # import model
from tensorboardX import SummaryWriter


def main():
    num_classes = 102
    model_name = 'Medmamba'
    batch_size = 8  # 32|  4(2681M)
    epochs = 100
    output_folder = r'output'
    date_format = '%Y-%m-%d_%H-%M-%S'  # 日期格式, 比如: 2021-07-01_12-00-00
    current_time = time.strftime(date_format, time.localtime())
    save_path = os.path.join(output_folder, current_time)
    os.makedirs(save_path, exist_ok=True)
    pth_path = os.path.join(save_path, model_name + '.pth')
    log_dir = os.path.join(save_path, 'logs')
    tb_writer = SummaryWriter(log_dir=log_dir)
    base_config_path = 'user/config.json'
    class_indices_path = 'user/class_indices.json'
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = json5.load(f)
    train_set = os.path.join(base_config['dataset'], 'train')
    val_set = os.path.join(base_config['dataset'], 'val')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    train_dataset = datasets.ImageFolder(root=train_set, transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open(class_indices_path, 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=val_set, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = medmamba(num_classes=num_classes)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_total = 0
        train_correct = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

            # Calculate train accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.to(device)).sum().item()

        train_accuracy = train_correct / train_total
        train_loss = running_loss / train_steps

        # Write train loss and accuracy to TensorBoard
        tb_writer.add_scalar('Train/Loss', train_loss, epoch)
        tb_writer.add_scalar('Train/Accuracy', train_accuracy, epoch)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss_val = loss_function(outputs, val_labels.to(device))
                val_running_loss += loss_val.item()
                predict_y = torch.max(outputs, dim=1)[1]
                val_total += val_labels.size(0)
                # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_correct += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accuracy = val_correct / val_total
        val_loss = val_running_loss / train_steps
        tb_writer.add_scalar('Val/Loss', val_loss, epoch)
        tb_writer.add_scalar('Val/Accuracy', val_accuracy, epoch)
        print('[epoch %d] train_acc: %.3f  train_loss: %.3f  val_acc: %.3f  val_loss: %.3f' % (epoch + 1, train_accuracy, train_loss, val_accuracy, val_loss))

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), pth_path)

    print('Finished Training')
    tb_writer.close()


if __name__ == '__main__':
    main()
    # print("This is Medmamba train.py")
