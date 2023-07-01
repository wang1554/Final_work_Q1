# eval.py
import torch,argparse
from torchvision.datasets import CIFAR10
import config
from torch import nn
import torchvision
from torchvision import  transforms
net = torchvision.models.resnet18(pretrained=True)
net.fc = nn.Linear(512, 10)
load_path=config.pre_model
print(load_path)
load_weight=torch.load(load_path)
net.load_state_dict(load_weight)
# transforms.Resize(size):将图片的短边缩放成size的比例，然后长边也跟着缩放，使得缩放后的图片相对于原图的长宽比不变
# transforms.CenterCrop(size):从图像的中心位置裁剪指定大小的图像
# ToTensor():将图像由PIL转换为Tensor
# transform.Normalize():把0-1变换到(-1,1)
# image = (image - mean) / std
# 其中mean和std分别通过(0.5, 0.5, 0.5)和(0.2, 0.2, 0.2)进行指定。原来的0-1最小值0则变成(0-0.5)/0.5=-1，而最大值1则变成(1-0.5)/0.5=1
trans = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.2, 0.2, 0.2))
])


def eval(args):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    eval_dataset=CIFAR10(root='dataset', train=False, transform=trans, download=True)
    eval_data=torch.utils.data.DataLoader(eval_dataset,batch_size=args.batch_size, shuffle=False, num_workers=16, )

    model=net.to(DEVICE)
    #model.load_state_dict(torch.load(config.pre_model, map_location='cpu'), strict=False)

    # total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(eval_data)
    total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0

    model.eval()
    with torch.no_grad():
        print("batch", " "*1, "top1 acc", " "*1,"top5 acc" )
        for batch, (data, target) in enumerate(eval_data):
            data, target = data.to(DEVICE) ,target.to(DEVICE)
            pred=model(data)

            total_num += data.size(0)
            prediction = torch.argsort(pred, dim=-1, descending=True)
            top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            top5_acc = torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_1 += top1_acc
            total_correct_5 += top5_acc

            print("  {:02}  ".format(batch+1)," {:02.3f}%  ".format(top1_acc / data.size(0) * 100),"{:02.3f}%  ".format(top5_acc / data.size(0) * 100))

        print("all eval dataset:","top1 acc: {:02.3f}%".format(total_correct_1 / total_num * 100), "top5 acc:{:02.3f}%".format(total_correct_5 / total_num * 100))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test SimCLR')
    parser.add_argument('--batch_size', default=512, type=int, help='')

    args = parser.parse_args()
    eval(args)
