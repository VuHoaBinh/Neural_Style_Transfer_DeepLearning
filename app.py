import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from flask import Flask, request, render_template, send_file
import os
import io

app = Flask(__name__)

# install tool
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# size image 512
imsize = 512 if torch.cuda.is_available() else 256

# save image tested and assess it during many steps of model
UPLOAD_FOLDER = 'upload'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load image import 
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def image_loader(image):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, io.BytesIO):
        image = Image.open(image).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Chuẩn hóa ngược
unloader = transforms.ToPILImage()

# Tải mô hình VGG19
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Chuẩn hóa đầu vào cho VGG
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Content Loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

# Style Loss
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Lấy các layer của mô hình
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, content_layers, style_layers):
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    
    content_losses = []
    style_losses = []
    
    model = nn.Sequential(normalization)
    
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
            
        model.add_module(name, layer)
        
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
            
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
            
    model = model[:(i + 1)]
    
    return model, content_losses, style_losses

# Hàm chạy style transfer
def run_style_transfer(cnn, normalization_mean, normalization_std,
                      content_img, style_img, input_img, num_steps=300,
                      style_weight=1000000, content_weight=1):
    model, content_losses, style_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img,
        content_layers=['conv_4'],
        style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    )
    
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    
    optimizer = optim.LBFGS([input_img])
    
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
                
            optimizer.zero_grad()
            model(input_img)
            
            content_score = 0
            style_score = 0
            
            for cl in content_losses:
                content_score += cl.loss
            for sl in style_losses:
                style_score += sl.loss
                
            style_score *= style_weight
            content_score *= content_weight
            
            loss = style_score + content_score
            loss.backward()
            
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")
                
            return content_score + style_score
            
        optimizer.step(closure)
        
    with torch.no_grad():
        input_img.clamp_(0, 1)
        
    return input_img

# API
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Lấy hình ảnh và tham số
        content_file = request.files['content_image']
        style_file = request.files['style_image']
        style_weight = int(request.form.get('style_weight', 1000000))
        content_weight = int(request.form.get('content_weight', 1))
        num_steps = int(request.form.get('num_steps', 300))
        
        if content_file and style_file:
            # Lưu hình ảnh tạm thời
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg')
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], 'style.jpg')
            content_file.save(content_path)
            style_file.save(style_path)
            
            # Chạy style transfer
            content_img = image_loader(content_path)
            style_img = image_loader(style_path)
            input_img = content_img.clone()
            
            output_tensor = run_style_transfer(
                cnn, cnn_normalization_mean, cnn_normalization_std,
                content_img, style_img, input_img,
                num_steps=num_steps, style_weight=style_weight, content_weight=content_weight
            )
            
            # Lưu kết quả
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
            output_img = unloader(output_tensor.squeeze(0).cpu())
            output_img.save(output_path)
            
            return render_template('index.html', output_image='uploads/output.jpg')
    
    return render_template('index.html', output_image=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)