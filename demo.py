import torch
from mobilenetv3 import mobilenetv3_small,mobilenetv3_large
from PIL import Image
from torchvision.transforms import transforms
from torch.nn import functional as F
test_trans = transforms.Compose([transforms.Resize((224, 224)), # 首先需resize成跟训练集图像一样的大小
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

net = mobilenetv3_small(width_mult=0.5)
ckpt = torch.load("./pretrain/mobilenetv3smallx0.5_Top1_55.94.pth",map_location=torch.device('cpu'))['state_dict']

#newckpt = {k[7:]:v for k,v in ckpt.items()}
net.load_state_dict(newckpt)
print(net)
#torch.save(net.state_dict(),"mobilenetv3smallx0.5_Top1_55.94.pth")

image = Image.open("./images/3a859c02856516d02d3c1d62b1fc491d.jpeg")
img = test_trans(image)
img = torch.unsqueeze(img, dim = 0)
net.eval()
out = net(img)
out = F.softmax(out)
print("Pred Class : ",out.argmax().item(),"Confidenc : ",out.max().item())
