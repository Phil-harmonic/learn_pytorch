import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from model import AlexNet

def predict(image_path, weight_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # assert os.path.exists(image_path), "file {} does not exist".format(image_path)
    # img = Image.open(image_path)
    img = image_path
    # plt.imshow(img)
    img = data_transform(img)
    #把维度扩展为[N, C, H, W]
    img = torch.unsqueeze(img, dim=0)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file {} does not exist".format(json_path)

    # with open(json_path, "r") as json_file:
    #     class_dict = json.load(json_file)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    #载入模型，开始训练
    model = AlexNet(num_classes=5).to(device)

    assert os.path.exists(weight_path), "file {} does not eixst".format(weight_path)
    model.load_state_dict(torch.load(weight_path))

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device)))
        # output = model(img)
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())

    # plt.title(print_res)
    # print(print_res)
    # plt.show()

    return predict_cla, class_indict[str(predict_cla)]

# if __name__ == "__main__":
#     predict(image_path="./sunflower.jpg", weight_path="./AlexNet.pth")
