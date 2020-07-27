import os, glob, datetime, pickle
from PIL import Image
import random
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Input images information
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = BASE_DIR +"/labels/Train"
TEST_DIR = BASE_DIR +"/labels/Test"
IMAGE_CLASSES = ["IR", "MT", "SP", "PG", "OV"]

## Parameter
SEED = 0
RATE_TEST = 0.2                   # rate of test data when splitting
img_height, img_width = 256, 256  # height and width of image
num_channel = 3                   # number of channels of image
BATCH_SIZE = 16                   # number of data points in each batch
NUM_EPOCHS = 20                   # times to run the model on complete data
LATENT_DIM = 64                   # latent vector dimension
LEARN_RATE = 1e-3                 # learning rate

## CNN model name
CNN_MODEL = "Inception"           # CNN_module ; "Inception" or "SimpleConv" are available
CLASS_SIZE = len(IMAGE_CLASSES)   # number of classes in the data

# Generate images
NUM_GENERATION = 9               # number of images to be generated
EXPORT_DIR = BASE_DIR +'/Reportfiles'

'''
generate array from images which are divided into folders with class names
'''
def train_test_split_dir():
    for idx,image_class in enumerate(IMAGE_CLASSES):
        image_dir = BASE_DIR + "/labels/" + image_class
        move_train_dir = TRAIN_DIR + "/" + image_class
        move_test_dir = TEST_DIR + "/" + image_class
        try:
            os.makedirs(move_train_dir, exist_ok=False)
            os.makedirs(move_test_dir, exist_ok=False)
            files = glob.glob(image_dir+'/*.jpg')
            print(len(files))
            th = math.floor(len(files)*RATE_TEST)
            random.shuffle(files)
            #RATE_TESTで指定したファイルをtestディレクトリに移動させる
            for i in range(th):
                shutil.copy(files[i],move_test_dir)
            #残りすべてをtrainディレクトリに移動させる
            files = glob.glob(image_dir+'/*.jpg')
            for file in files:
                shutil.copy(file,move_train_dir)
        except FileExistsError:
            pass
        print('----{}を処理----'.format(image_class))

def onehot_encode(label):
    return torch.eye(CLASS_SIZE, device=DEVICE, dtype=torch.float32)[label]

def concat_image_label(input_image, label):
    B, C, H, W = input_image.shape
    label_to_image = onehot_encode(label).view(-1,CLASS_SIZE, 1, 1).expand(B, CLASS_SIZE, H, W)
    return torch.cat((input_image, label_to_image), dim=1)

def save_img(img, title):
    npimg = img.to('cpu').detach().numpy().copy()
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
    img_path = EXPORT_DIR + f'/generate_images_{title}.png'
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'Generate images: {title}')
    plt.savefig(img_path)


## Encoder
def create_encoder_single_conv(in_chs, out_chs, kernel):
    assert kernel % 2 == 1
    return nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2),
                         nn.BatchNorm2d(out_chs),
                         nn.ReLU(inplace=True))

class EncoderInceptionModuleSignle(nn.Module):
    def __init__(self, channels):
        assert channels % 2 == 0
        super().__init__()
        # put bottle-neck layers before convolution
        bn_ch = channels // 2
        self.bottleneck = create_encoder_single_conv(channels, bn_ch, 1)
        # bn -> Conv1, 3, 5
        self.conv1 = create_encoder_single_conv(bn_ch, channels, 1)
        self.conv3 = create_encoder_single_conv(bn_ch, channels, 3)
        self.conv5 = create_encoder_single_conv(bn_ch, channels, 5)
        self.conv7 = create_encoder_single_conv(bn_ch, channels, 7)
        # pool-proj(no-bottle neck)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        # Original inception is concatenation, but use simple addition instead
        bn = self.bottleneck(x)
        out = self.conv1(bn) + self.conv3(bn) + self.conv5(bn) + self.conv7(bn) + self.pool3(x) + self.pool5(x)
        return out

class EncoderModule(nn.Module):
    def __init__(self, chs, repeat_num, cnn_model):
        super().__init__()
        if cnn_model=="Inception":
            layers = [EncoderInceptionModuleSignle(chs) for i in range(repeat_num)]
        elif cnn_model=="SimpleConv":
            layers = [create_encoder_single_conv(chs, chs, 3) for i in range(repeat_num)]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)

class Encoder(nn.Module):
    def __init__(self, cnn_model, repeat_per_module):
        super().__init__()
        # stages
        self.upch1 = nn.Conv2d(num_channel+CLASS_SIZE, 32, kernel_size=1)
        self.stage1 = EncoderModule(32, repeat_per_module, cnn_model)
        self.upch2 = self._create_downsampling_module(32, 4)
        self.stage2 = EncoderModule(64, repeat_per_module, cnn_model)
        self.upch3 = self._create_downsampling_module(64, 4)
        self.stage3 = EncoderModule(128, repeat_per_module, cnn_model)
        self.upch4 = self._create_downsampling_module(128, 2)
        self.stage4 = EncoderModule(256, repeat_per_module, cnn_model)

    def _create_downsampling_module(self, input_channels, pooling_kenel):
        return nn.Sequential(
            nn.AvgPool2d(pooling_kenel),
            nn.Conv2d(input_channels, input_channels * 2, kernel_size=1),
            nn.BatchNorm2d(input_channels * 2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.stage1(self.upch1(x))
        out = self.stage2(self.upch2(out))
        out = self.stage3(self.upch3(out))
        out = self.stage4(self.upch4(out))
        out = F.avg_pool2d(out, 8)  # Global Average pooling
        return out.view(-1, 256)


## Decoder
def create_decoder_single_conv(in_chs, out_chs, kernel):
    assert kernel % 2 == 1
    return nn.Sequential(nn.ConvTranspose2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2),
                         nn.BatchNorm2d(out_chs),
                         nn.ReLU(inplace=True))


class DecoderInceptionModuleSingle(nn.Module):
    def __init__(self, channels):
        assert channels % 2 == 0
        super().__init__()
        # put bottle-neck layers before convolution
        bn_ch = channels // 4
        self.bottleneck = create_decoder_single_conv(channels, bn_ch, 1)
        # bn -> Conv1, 3, 5
        self.conv1 = create_decoder_single_conv(bn_ch, channels, 1)
        self.conv3 = create_decoder_single_conv(bn_ch, channels, 3)
        self.conv5 = create_decoder_single_conv(bn_ch, channels, 5)
        self.conv7 = create_decoder_single_conv(bn_ch, channels, 7)
        # pool-proj(no-bottle neck)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        # Original inception is concatenation, but use simple addition instead
        bn = self.bottleneck(x)
        out = self.conv1(bn) + self.conv3(bn) + self.conv5(bn) + self.conv7(bn) + self.pool3(x) + self.pool5(x)
        return out


class DecoderModule(nn.Module):
    def __init__(self, chs, repeat_num, cnn_model):
        super().__init__()
        if cnn_model=="Inception":
            layers = [DecoderInceptionModuleSingle(chs) for i in range(repeat_num)]
        elif cnn_model=="SimpleConv":
            layers = [create_decoder_single_conv(chs, chs, 3) for i in range(repeat_num)]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class Decoder(nn.Module):
    def __init__(self, cnn_model, repeat_per_module):
        super().__init__()
        # stages
        self.stage1 = DecoderModule(256, repeat_per_module, cnn_model)
        self.downch1 = self._create_upsampling_module(256, 2)
        self.stage2 = DecoderModule(128, repeat_per_module, cnn_model)
        self.downch2 = self._create_upsampling_module(128, 4)
        self.stage3 = DecoderModule(64, repeat_per_module, cnn_model)
        self.downch3 = self._create_upsampling_module(64, 4)
        self.stage4 = DecoderModule(32, repeat_per_module, cnn_model)
        self.last = nn.ConvTranspose2d(32, num_channel, kernel_size=1)

    def _create_upsampling_module(self, input_channels, pooling_kenel):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=pooling_kenel, stride=pooling_kenel),
            nn.BatchNorm2d(input_channels // 2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        out = F.interpolate(x.view(-1, 256, 1, 1), scale_factor=8)
        out = self.downch1(self.stage1(out))
        out = self.downch2(self.stage2(out))
        out = self.downch3(self.stage3(out))
        out = self.stage4(out)
        return torch.sigmoid(self.last(out))


## VAE
class VAE(nn.Module):
    def __init__(self, cnn_model, repeat_per_block):
        super().__init__()

        # # latent features
        self.n_latent_features = LATENT_DIM

        # Encoder
        self.encoder = Encoder(cnn_model, repeat_per_block)
        # Middle
        self.fc_mu = nn.Linear(256, self.n_latent_features)
        self.fc_logvar = nn.Linear(256, self.n_latent_features)
        self.fc_rep = nn.Linear(self.n_latent_features+CLASS_SIZE, 256)
        # Decoder
        self.decoder = Decoder(cnn_model, repeat_per_block)

        # model_name
        flag = cnn_model
        self.model_name = f"{flag}_epoch{NUM_EPOCHS}"

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(DEVICE)
        z = mu + std * esp
        return z

    def _bottleneck(self, h):
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, inputs, labels):
        # Encoder
        h = self.encoder(inputs)
        # Bottle-neck
        z, mu, logvar = self._bottleneck(h)
        # decoder
        latent_inputs = torch.empty((z.shape[0], self.n_latent_features + CLASS_SIZE), device = DEVICE)
        latent_inputs[:, :self.n_latent_features] = z
        latent_inputs[:, self.n_latent_features:] = onehot_encode(labels)
        latent_inputs = self.fc_rep(latent_inputs)
        decoded_inputs = self.decoder(latent_inputs)
        d = concat_image_label(decoded_inputs, labels)
        return d, mu, logvar

    def loss_function(self, recon_x, inputs, mu, logvar):
        # https://arxiv.org/abs/1312.6114 (Appendix B)
        BCE = F.binary_cross_entropy(recon_x, inputs, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def GeneratedImageSample(self):
        for idx, image_class in enumerate(IMAGE_CLASSES):
            class_label = onehot_encode(idx).to(DEVICE).unsqueeze(dim=0)
            for i in range(NUM_GENERATION):
                random_latent_dim = torch.randn(self.n_latent_features, device=DEVICE).unsqueeze(dim=0)
                z = torch.cat((random_latent_dim, class_label), dim=1)
                z = self.fc_rep(z)
                sample_image = self.decoder(z)
                if i == 0:
                    images = sample_image
                else:
                    images = torch.cat((images, sample_image), dim=0)
            export_img = make_grid(images, nrow=3, padding=2)
            save_img(export_img,image_class)
            print(f'Generate images: {image_class}')


def train(vae, loader, optimizer, history, epoch):
    vae.train()
    print(f"\nEpoch: {epoch + 1:d} {datetime.datetime.now()}")
    samples_cnt = 0
    train_loss = 0
    for batch_idx, data in enumerate(loader):
        images = data[0].to(DEVICE)
        labels = data[1].to(DEVICE)
        inputs = concat_image_label(images, labels)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(inputs, labels)

        loss = vae.loss_function(recon_batch, inputs, mu, logvar)
        loss.backward()
        optimizer.step()

        samples_cnt += images.size(0)
        train_loss += loss.item()

    avg_train_loss = train_loss / samples_cnt
    history["train_loss"].append(avg_train_loss)
    print('Epoch [{}/{}], Loss: {loss: .4f}'
        .format(epoch + 1, NUM_EPOCHS, loss=avg_train_loss))


def test(vae, loader, history, epoch):
    vae.eval()
    samples_cnt = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            images = data[0].to(DEVICE)
            labels = data[1].to(DEVICE)
            inputs = concat_image_label(images, labels)
            recon_batch, mu, logvar = vae(inputs, labels)
            loss = vae.loss_function(recon_batch, inputs, mu, logvar)
            samples_cnt += images.size(0)
            test_loss += loss.item()

    avg_test_loss = test_loss / samples_cnt
    history["test_loss"].append(avg_test_loss)
    print('Epoch [{}/{}], Validation loss: {test_loss: .4f}'
        .format(epoch + 1, NUM_EPOCHS, test_loss=avg_test_loss))


def results(epoch, train_loss_list, test_loss_list):
    plt.figure()
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
    img_path = EXPORT_DIR + '/training_val_loss.png'
    plt.plot(range(epoch), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(epoch), test_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()
    plt.savefig(img_path)

def report_generates_images(vae):
    vae.eval()
    vae.GeneratedImageSample()
    print('Complete create generated images')


def save_model(model):
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
    filename = EXPORT_DIR + '/finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))


def main():
    # split train / test images
    train_test_split_dir()
    # load data
    data_transform = transforms.Compose([transforms.Resize(img_height), transforms.ToTensor()])
    train_data = torchvision.datasets.ImageFolder(root=TRAIN_DIR, transform=data_transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = torchvision.datasets.ImageFolder(root=TEST_DIR, transform=data_transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    print('Complete loading train / test images')
    # model
    net = VAE(CNN_MODEL, 1)
    # init
    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
    net.to(DEVICE)
    # history
    history = {"train_loss": [], "test_loss": []}
    # train
    for i in range(NUM_EPOCHS):
        train(net, train_loader, optimizer, history, i)
        test(net, test_loader, history, i)
    report_generates_images(net)
    results(NUM_EPOCHS, history["train_loss"], history["test_loss"])
    # save results
    save_model(net)

if __name__ == "__main__":
    main()