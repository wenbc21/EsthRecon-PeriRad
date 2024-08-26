import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model.model_zoo import model_dict
import glob

from sklearn import metrics
import matplotlib.pylab as plt
import numpy as np


def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.5, axis_labels=None, s=""):
    # 利用sklearn中的函数生成混淆矩阵并归一化
    
    y_p = []
    for i in y_pred :
        if i > thresh :
            y_p.append(1)
        else: 
            y_p.append(0)
    y_pred = y_p
    
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵 
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = ["N", "Y"]
    plt.xticks(num_local, axis_labels)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    # 显示
    plt.savefig(s)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 2
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize((img_size, img_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
         transforms.Normalize([0.29204324, 0.29204324, 0.29204324], [0.29269517, 0.29269517, 0.29269517])])

    y_true = []
    y_pred = []
    for i in range(4) :
        y_true.append(1.0)
    for i in range(16) :
        y_true.append(0.0)

    # load image
    label = 'Y'
    img_dir = f"dataset/AT_new"
    predict_t = 0
    predict_n = 0
    assert os.path.exists(img_dir), "file: '{}' dose not exist.".format(img_dir)
    img_files = [i for i in glob.glob(os.path.join(img_dir, '*.jpg'))]
    img_files.sort()
    for img_path in img_files :
        img = Image.open(img_path)
        # plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = 'dataset/class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = model_dict["resnet50"](num_classes=num_classes).to(device)
        # load model weights
        model_weight_path = "weights/resnet50_best.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        
        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                             predict[predict_cla].numpy())
        # plt.title(print_res)

        if predict[0].numpy() < predict[1].numpy() :
            print("label: {} img_path: {}  class: {:10}   prob: {:.3}".format(label,img_path.split('/')[-1], class_indict[str(1)],
                                                    predict[1].numpy()))
            if class_indict[str(1)] == label :
                predict_t += 1
            else :
                predict_n += 1
        else:
            print("label: {} img_path: {}  class: {:10}   prob: {:.3}".format(label,img_path.split('/')[-1], class_indict[str(0)],
                                                    predict[0].numpy()))
            if class_indict[str(0)] == label :
                predict_t += 1
            else :
                predict_n += 1
        # plt.show()
        y_pred.append(predict[1].numpy())

    print(f"TP: {predict_t}    FN: {predict_n}")


    # # load image
    # label = 'N'
    # img_dir = f"dataset/Task3clsAug/test/{label}"
    # predict_t = 0
    # predict_n = 0
    # assert os.path.exists(img_dir), "file: '{}' dose not exist.".format(img_dir)
    # img_files = [i for i in glob.glob(os.path.join(img_dir, '*.jpg'))]
    # for img_path in img_files :
    #     img = Image.open(img_path)
    #     # plt.imshow(img)
    #     # [N, C, H, W]
    #     img = data_transform(img)
    #     # expand batch dimension
    #     img = torch.unsqueeze(img, dim=0)

    #     # read class_indict
    #     json_path = 'cls_model/ConvNeXt/class_indices.json'
    #     assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    #     with open(json_path, "r") as f:
    #         class_indict = json.load(f)

    #     # create model
    #     model = convnext_large(num_classes=num_classes).to(device)
    #     # load model weights
    #     model_weight_path = "cls_model/ConvNeXt/weights/convnext_large_best.pth"
    #     model.load_state_dict(torch.load(model_weight_path, map_location=device))
    #     model.eval()
    #     with torch.no_grad():
    #         # predict class
    #         output = torch.squeeze(model(img.to(device))).cpu()
    #         predict = torch.softmax(output, dim=0)
    #         predict_cla = torch.argmax(predict).numpy()
        
    #     # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #     #                                             predict[predict_cla].numpy())
    #     # plt.title(print_res)

    #     if predict[0].numpy() < predict[1].numpy() :
    #         print("label: {}   class: {:10}   prob: {:.3}".format(label, class_indict[str(1)],
    #                                                 predict[1].numpy()))
    #         if class_indict[str(1)] == label :
    #             predict_t += 1
    #         else :
    #             predict_n += 1
    #     else:
    #         print("label: {}   class: {:10}   prob: {:.3}".format(label, class_indict[str(0)],
    #                                                 predict[0].numpy()))
    #         if class_indict[str(0)] == label :
    #             predict_t += 1
    #         else :
    #             predict_n += 1
    #     # plt.show()
    #     y_pred.append(predict[1].numpy())

    # print(f"TN: {predict_t}    FP: {predict_n}")

    # # print(y_true)
    # # print(y_pred)

    # #roc
    # fpr1, tpr1, thresholds = metrics.roc_curve(y_true, y_pred)
    # roc_auc1 = metrics.auc(fpr1, tpr1)  # the value of roc_auc1
    # print(roc_auc1)
    # plt.plot(fpr1, tpr1, 'r', label='AUROC = %0.4f' % roc_auc1)
    # plt.plot([0.0, 1.0], [0.0, 1.0], 'gray', linestyle='--')
    # plt.legend(loc='lower right')
    # plt.xlim([-0.05, 1.05])  # the range of x-axis
    # plt.ylim([-0.05, 1.05])  # the range of y-axis
    # plt.xlabel('False Positive Rate')  # the name of x-axis
    # plt.ylabel('True Positive Rate')  # the name of y-axis
    # plt.title('ConvNeXt-large AUROC')  # the title of figure
    # plt.savefig("./cls_model/ConvNeXt/results/ConvNeXt_large_AUROC.png")
    # # plt.show()
    # plt.close()

    # #prc
    # precision1, recall1, _ = metrics.precision_recall_curve(y_true, y_pred)
    # aupr1 = metrics.auc(recall1, precision1)  # the value of roc_auc1
    # print(aupr1)
    # plt.plot(recall1, precision1, 'b', label='AUPRC = %0.4f' % aupr1)
    # plt.plot([0.0, 1.0], [1.0, 0.0], 'gray', linestyle='--')
    # plt.legend(loc='lower left')
    # plt.xlim([-0.05, 1.05])  # the range of x-axis
    # plt.ylim([-0.05, 1.05])  # the range of y-axis
    # plt.xlabel('Recall')  # the name of x-axis
    # plt.ylabel('Precision')  # the name of y-axis
    # plt.title('ConvNeXt-large AUPRC')  # the title of figure
    # plt.savefig("./cls_model/ConvNeXt/results/ConvNeXt_large_AUPRC.png")
    # # plt.show()
    # plt.close()

    # plot_matrix(y_true, y_pred, [0, 1], title=f"ConvNeXt-large Confusion Matrix", thresh=0.5, axis_labels=None, s=f"./cls_model/ConvNeXt/results/ConvNeXt_large_Confusion_Matrix.png")


if __name__ == '__main__':
    main()
