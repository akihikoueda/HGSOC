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
from inception3 import InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux, BasicConv2d
import warnings
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Input images information
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = BASE_DIR + "/labels/Train"
TEST_DIR = BASE_DIR + "/labels/Test"
IMAGE_CLASSES = ["IR", "MT", "SP", "PG", "OV"]

## Parameter
RATE_TEST = 0.2                   # rate of test data when splitting
img_height, img_width = 299, 299  # height and width of image
num_channel = 3                   # number of channels of image
BATCH_SIZE = 16                   # number of data points in each batch
NUM_EPOCHS = 50                   # times to run the model on complete data
LATENT_DIM = 64                   # latent vector dimension
LEARN_RATE = 1e-6                 # learning rate
op_encoder = 256

## CNN model name
CNN_MODEL = "Inception"           # CNN_module ; "Inception" or "SimpleConv" are available
CLASS_SIZE = len(IMAGE_CLASSES)   # number of classes in the data

# Generate images
NUM_GENERATION = 9               # number of images to be generated
EXPORT_DIR = BASE_DIR +'/Reportfiles07'

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
            # RATE_TESTで指定したファイルをtestディレクトリに移動させる
            for i in range(th):
                shutil.copy(files[i],move_test_dir)
            # 残りすべてをtrainディレクトリに移動させる
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
    label = label/CLASS_SIZE
    label_to_image = label.view(-1,1,1,1).expand(B,1,H,W).float()
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
class Encoder(nn.Module):

    def __init__(self, num_classes=op_encoder, aux_logits=True, transform_input=False,
                 inception_blocks=None, init_weights=None):
        super().__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        if init_weights is None:
            warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.Conv2d_00_1x1 = conv_block(4, 3, kernel_size=1)
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        # N x 4 x 299 x 299
        x = self.Conv2d_00_1x1(x)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1  1
        #x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        
        # N x 2048
        
        out = x.view(-1,2048)
        # N x 1000 (num_classes)
        return out


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
        self.downch1 = self._create_upsampling_module(256, 3, 0)
        self.stage2 = DecoderModule(128, repeat_per_module, cnn_model)
        self.downch2 = self._create_upsampling_module(128, 3, 2)
        self.stage3 = DecoderModule(64, repeat_per_module, cnn_model)
        self.downch3 = self._create_upsampling_module(64, 4, 3)
        self.stage4 = DecoderModule(32, repeat_per_module, cnn_model)
        self.last = nn.ConvTranspose2d(32, num_channel, kernel_size=1)

    def _create_upsampling_module(self, input_channels, pooling_kenel,outpadding):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=pooling_kenel, stride=pooling_kenel, output_padding=outpadding),
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
        self.encoder = Encoder(op_encoder)
        # Middle
        self.fc_mu = nn.Linear(2048, self.n_latent_features)
        self.fc_logvar = nn.Linear(2048, self.n_latent_features)
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
    data_transform = transforms.Compose([transforms.Resize(size=(img_height,img_width)), transforms.ToTensor()])
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
