import torch
from torchvision.models import resnet34

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = resnet34(pretrained=True)
model = model.to(device)


class SaveOutput:
    def __init__(self):
        self.outputs = []
        self.inputs = []

    def __call__(self, module, module_in, module_out):
        print(module)
        print(module_in)
        print(module_out)
        print("=====")
        self.inputs.append(module_in)
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []
        self.inputs = []

def forward(module, module_in, module_out):
    print("forward:", end=" ")
    # print(module)
    # print(module_in[0].shape)
    # print(module_out[0].shape)
    # print("===")

def backward(module, module_in, module_out):
    print("backward:", end=" ")
    print(module)
    print(module_in[0].shape)
    print(module_out[0].shape)
    # print("===")
    module_input = module_in
    mask = torch.rand(module_in[0].shape).cuda()
    module_input = torch.mul(module_in[0], mask), module_in[1]
    return module_input



for layer in model.modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d) and layer is not None:
        layer.register_forward_hook(forward)
        handle = layer.register_backward_hook(backward)

from PIL import Image
from torchvision import transforms as T

img = Image.open('image/cat.jpeg')
transform = T.Compose([T.Resize((224,224)),
                       T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.485, 0.456, 0.406],)
                      ])
x = transform(img).unsqueeze(0).to(device)
out = model(x)
# criterion = torch.nn.CrossEntropyLoss()
loss = out.sum()
print("**************")
loss.backward()

