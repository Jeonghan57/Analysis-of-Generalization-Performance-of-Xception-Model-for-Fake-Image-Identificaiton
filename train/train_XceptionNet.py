# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:33:38 2021

@author: Jeonghan
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import tqdm

from torchvision.datasets import ImageFolder
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #gpu 사용 가능?


# ImageFolder 함수를 사용해서 Dataset 작성
train_imgs = ImageFolder(f"./Datafolder/FFHQ/train(StyleGAN2)/",
                         transform=transforms.Compose([
                                 transforms.ToTensor()]
))
test_imgs = ImageFolder(f"./Datafolder/LSUN/test(horse_StyleGAN2)/",
                         transform=transforms.Compose([
                                 transforms.ToTensor()]
))


# DataLoader 작성
train_loader = DataLoader(
        train_imgs, batch_size=16, shuffle=True)
test_loader = DataLoader(
        test_imgs, batch_size=16, shuffle=False)

"""
# 모델 불러오기 및 정의
from EfficientNet import efficientnet_b4
net = efficientnet_b4(num_classes=2)
"""

def patch_processing(x):
    kc, kh, kw = 3, 64, 64 # kernel size
    dc, dh ,dw = 3, 64, 64 # stride
    
    patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(patches.size(0), -1, kc, kh, kw)
    
    return patches

# 모델 불러오기 및 정의
from XceptionNet import Xception
net = Xception(num_classes=2)

def test_net(net, data_loader, device="cpu"):
    # Dropout 및 BatchNorm을 무효화
    net.eval()
    ys = []
    ypreds = []
    for x ,y in data_loader:
        # to 메서드로 계산을 실행할 디바이스로 전송
        x = x.to(device)
        y = y.to(device)
        # 확률이 가장 큰 분류를 예측
        # 여기선 forward(추론) 계산이 전부이므로 자동 미분에
        # 필요한 처리는 off로 설정해서 불필요한 계산을 제한다
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)
        
    # 미니 배치 단위의 예측 결과 등을 하나로 묶는다
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    # 예측 정확도 계산
    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()

def train_net(net, train_loader, test_loader,
              optimizer_cls=optim.Adam,
              loss_fn=nn.CrossEntropyLoss(),
              n_iter=10, device="cpu"):
    train_losses = []
    train_acc = []
    val_acc = []
    optimizer = optimizer_cls(net.parameters())
    
    for epoch in range(n_iter):
        running_loss = 0.001
        # 신경망을 훈련 모드로 설정
        net.train()
        n = 0
        n_acc = 0
        # 시간이 많이 걸리므로 tqdm을 사용해서 진행 바를 표시
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader),
                                    total=len(train_loader)):
            xx = patch_processing(xx)
            xx = xx.to(device)
            yy = yy.to(device)
            h = net(xx)
            loss = loss_fn(h,yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)
            _, y_pred = h.max(1)
            n_acc += (yy == y_pred).float().sum().item()
        train_losses.append(running_loss / i)
        #훈련 데이터의 예측 정확도
        train_acc.append(n_acc / n)
        
        # 검증 데이터의 예측 정확도
        val_acc.append(test_net(net, test_loader, device))
        
        # epoch의 결과 표시
        print("\nepoch {0} --> loss : {1} / train_acc : {2}% / val_acc : {3}% \n"
              .format(epoch, train_losses[-1], train_acc[-1]*100, val_acc[-1]*100, flush=True))

        torch.save(net.state_dict(), f"./result/Patch Experiments/Face(StyleGAN2)(64)/epoch_" + str(epoch) + ".pth")
    

def Evaluate_Networks(Net):
    save_path = "./result/Generalization Experiments/Add(XceptionNet_dropout 0.5)/"
    # data
    Net.load_state_dict(torch.load(save_path + "epoch(Add_10)_49.pth"), strict=False)
    Net = Net.to(device).eval()
    
    test_data = test_loader
    
    # Test
    ys = []
    ypreds = []
    for X, Y in tqdm.tqdm(test_data):
        X = X.to(device)
        Y = Y.to(device)

        with torch.no_grad():
            # Value, Indices >> Get Indices
            _, y_pred = Net(X).max(1)
            ys.append(Y)
            ypreds.append(y_pred)

        
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)

    acc_real = (ys[2000:] == ypreds[2000:]).float().sum() / len(ys[2000:])
    acc_fake = (ys[:2000] == ypreds[:2000]).float().sum() / len(ys[:2000])
    acc = (ys == ypreds).float().sum() / len(ys)


    print('\n-----------------------------------------')
    print('Real Accuracy : ', acc_real.item())
    print('Fake Accuracy : ', acc_fake.item())
    print('Total AVG : ', acc.item())


# 신경망의 모든 파라미터를 GPU로 전송
net.to("cuda:0")

# 훈련 실행
# train_net(net, train_loader, test_loader, n_iter=50, device="cuda:0")

# Evaluation
Evaluate_Networks(net)