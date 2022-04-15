import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets, utils
import torch.optim as optim
from tqdm import tqdm
import os
import json
from model import AlexNet
import time
from torch.utils.tensorboard import SummaryWriter

def train():
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    tb_writer = SummaryWriter(log_dir="runs")
    print("using {} device".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "learn_pytorch","Datasets", "data_set", "flower_data")
    assert os.path.exists(image_path), "{} path does not exist".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("using {} dataloder workers every process".format(nw))

    train_loder = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle = True,
                                             num_workers = 0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loder = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size,
                                                shuffle = True,
                                                num_workers = 0)

    print("using {} images for training, {} images for validation".format(train_num, val_num))

    #载入网络 开始训练
    net = AlexNet(num_classes=5, init_weights=True)
    net.to(device)
    print(net)

    init_img = torch.zeros((1, 3, 224, 224), device=device)
    tb_writer.add_graph(net, init_img)

    loss_founction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0002)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epochs = 100
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loder)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        lr = optimizer.param_groups[0]["lr"]
        t1 = time.perf_counter()
        # train_bar = tqdm(train_loder)
        for step, data in enumerate(train_loder, start=0):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_founction(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # train_bar.desc = "\rtrain epoch[{}/{} loss: {:.3f}]".format(epoch+1, epochs, loss)
            rate = (step + 1) / len(train_loder)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\repoch[{}/{}] train_loss: {:^3.0f}%[{}->{}]  {:.3f}".format(epoch+1, epochs, int(rate * 100), a, b, loss), end="")
            total_time = (time.perf_counter() - t1) / 60
            tb_writer.add_scalar("train_loss/batch_loss", loss.item(), step)
        # print()
        print("    total time:{:.3f}min".format(total_time))

        net.eval()
        acc = 0.0
        with torch.no_grad():
            # val_bar = tqdm(validate_loder)
            for val_data in validate_loder:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim = 1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()#对预测正确的个数进行求和
                # val_bar.desc = "val epoch[{}/{} accurate_number: {:.3f}]".format(epoch+1, epochs, acc)
        val_accurate = acc / val_num
        running_loss /= train_steps
        print("           train_loss: {:.3f} val_accuracy: {:.3f}".format(running_loss/train_steps, val_accurate))

        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        #     torch.save(net.state_dict(), save_path)
        tb_writer.add_scalar("train_loss/epoch_loss", running_loss, epoch)
        tb_writer.add_scalar("val_accuracy/val_acc", val_accurate, epoch)
        tb_writer.add_scalar("learning_rate", lr, epoch)

        tb_writer.add_histogram("conv1", net.features[0].weight, epoch)

    print("Finished Training")



if __name__ == "__main__":
    train()