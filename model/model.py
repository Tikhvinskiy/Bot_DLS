import requests
from data import config
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import functional
from torchvision.utils import save_image
import copy


def gram_matrix(input):
    batch_size, h, w, f_map_num = input.size()
    features = input.view(batch_size * h, w * f_map_num)
    G = torch.mm(features, features.t())
    return G.div(batch_size * h * w * f_map_num)


class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class Transformation:

    def __init__(self, style_power, image_size, chat_id, epoch):
        self.chat_id = chat_id
        self.num_steps = epoch
        self.image_size = image_size
        self.style_power = style_power
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unloader = transforms.ToPILImage()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        # torch.save(models.vgg19(pretrained=True).features.to(self.device).eval(), './model/vgg19.pt')
        self.cnn = torch.load('./model/vgg19.pt').to(self.device)

    def processing(self, content, style):
        size_of_original = Image.open(content).size
        content_img = self.image_loader(content, content=True)
        size1, size2 = content_img.shape[2], content_img.shape[3]
        style_img = self.image_loader(style, size1, size2, content=False)

        input_img = content_img.clone()
        output = self.run_style_transfer(self.cnn,
                                         self.cnn_normalization_mean,
                                         self.cnn_normalization_std,
                                         content_img, style_img, input_img,
                                         style_weight=self.image_size ** self.style_power // 10)
        out_size = size_of_original[0]
        if size_of_original[0] > size1 or size_of_original[1] > size2:
            output = functional.resize(output, out_size, antialias=True)
            t = transforms.transforms.GaussianBlur(11, sigma=(0.1, 2.0))
            output = t(output)
        return save_image(output, f"./photos/out{self.chat_id}.jpg")

    def image_loader(self, image_name, size1=None, size2=None, content=True):
        image = Image.open(image_name)
        if content:
            loader = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()])
        else:
            loader = transforms.Compose([
                transforms.Resize((size1, size2)),
                transforms.ToTensor()])
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=['conv_4'],
                                   style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):

        cnn = copy.deepcopy(cnn)
        normalization = Normalization(normalization_mean, normalization_std).to(self.device)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(self, cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=self.num_steps,
                           style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        print(f'Style_weight={style_weight}\nKernel={self.image_size}\nBuilding the style transfer model.. ')
        model, style_losses, content_losses = self.get_style_model_and_losses(cnn,
                                                                              normalization_mean, normalization_std,
                                                                              style_img, content_img)
        print('Get_style')
        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    request = f"https://api.telegram.org/bot{config.BOT_TOKEN}/sendMessage?chat_id={self.chat_id}&text=Сделанно {run[0]} шагов из {num_steps}"
                    try:
                        requests.get(request, timeout=1)
                    except Exception:
                        pass
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                return style_score + content_score

            optimizer.step(closure)
        input_img.data.clamp_(0, 1)
        return input_img
