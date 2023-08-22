import torch
import torch.nn as nn

from . import cnn_backbones


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()

        self.output_dim = cfg.model.text.embedding_dim
        # self.norm = cfg.model.norm
        self.norm = True

        model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
        self.model, self.feature_dim, self.interm_feature_dim = model_function(
            pretrained=cfg.model.vision.pretrained
        )

        self.global_embedder = nn.Linear(self.feature_dim, self.output_dim)
        self.local_embedder = nn.Conv2d(
            self.interm_feature_dim,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if cfg.model.vision.freeze_cnn:
            print("Freezing CNN model")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, get_local=False):

        if "resnet" or "resnext" in self.cfg.model.vision.model_name:

            global_ft, local_ft, x_, e1, e2, e3= self.resnet_forward(x, extract_features=True)
        elif "densenet" in self.cfg.model.vision.model_name:
            global_ft, local_ft = self.dense_forward(x, extract_features=True)

        if get_local:
            return global_ft, local_ft, x_, e1, e2, e3
        else:
            return global_ft

    def generate_embeddings(self, global_features, local_features):

        global_emb = self.global_embedder(global_features)
        local_emb = self.local_embedder(local_features)

        if self.norm is True:
            local_emb = local_emb / torch.norm(
                local_emb, 2, dim=1, keepdim=True
            ).expand_as(local_emb)
            global_emb = global_emb / torch.norm(
                global_emb, 2, dim=1, keepdim=True
            ).expand_as(global_emb)

        return global_emb, local_emb

    def resnet_forward(self, x, extract_features=False):


        x = nn.Upsample(size=(512, 512), mode="bilinear", align_corners=True)(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x_ = self.model.relu(x)
        x = self.model.maxpool(x_)

        e1 = self.model.layer1(x)  # (batch_size, 256, 256, 128)
        e2 = self.model.layer2(e1)  # (batch_size, 512, 128, 64)
        e3 = self.model.layer3(e2)  # (batch_size, 1024, 64, 32)
        local_features = e3
        e4 = self.model.layer4(e3)  # (batch_size, 2048, 32, 16)

        x = self.pool(e4)
        x = x.view(x.size(0), -1)

        return x, local_features, x_, e1, e2, local_features

    def densenet_forward(self, x, extract_features=False):
        pass

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)


class ImageDecoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.BN_enable = True

        filters = [64, 256, 512, 1024, 2048]


        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3] * 4, out_channels=filters[3],
                                   BN_enable=self.BN_enable)
        self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable)
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[1], BN_enable=self.BN_enable)
        self.decoder3 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[0], BN_enable=self.BN_enable)

        if self.BN_enable:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1),
                nn.Sigmoid()
                # nn.Tanh()
            )
            self.mfinal = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                nn.Sigmoid()
                # nn.Tanh()
            )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=output_channel, kernel_size=1),
                nn.Sigmoid()
                # nn.Tanh()
            )

    def forward(self, x_, e1, e2, e3, mt):

        center = self.center(e3)
        d2 = self.decoder1(torch.cat([center, e2], dim=1))
        # d2 = self.decoder1(torch.cat([e3, e2], dim=1))
        d3 = self.decoder2(torch.cat([d2, e1], dim=1))
        d4 = self.decoder3(torch.cat([d3, x_], dim=1))

        center1 = self.center(e3)
        d2_ = self.decoder1(torch.cat([center1, e2], dim=1))
        # d2 = self.decoder1(torch.cat([e3, e2], dim=1))
        d3_ = self.decoder2(torch.cat([d2_, e1], dim=1))
        d4_ = self.decoder3(torch.cat([d3_, x_], dim=1))

        if mt:
            return self.final(d4), self.mfinal(d4_)
        else:
            return self.final(d4), self.final(d4_)


class PretrainedImageClassifier(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        num_cls: int,
        feature_dim: int,
        freeze_encoder: bool = True,
    ):
        super(PretrainedImageClassifier, self).__init__()
        self.img_encoder = image_encoder
        self.classifier = nn.Linear(feature_dim, num_cls)
        if freeze_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred


class ImageClassifier(nn.Module):
    def __init__(self, cfg, image_encoder=None):
        super(ImageClassifier, self).__init__()

        model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
        self.img_encoder, self.feature_dim, _ = model_function(
            pretrained=cfg.model.vision.pretrained
        )

        self.classifier = nn.Linear(self.feature_dim, cfg.model.vision.num_targets)

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred
class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle', BN_enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1,
                              bias=False)

        if self.BN_enable:
            self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        if self.upsample_mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels,
                                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode == 'pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)
        if self.BN_enable:
            self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.BN_enable:
            x = self.norm1(x)
        x = self.relu1(x)
        x = self.upsample(x)
        if self.BN_enable:
            x = self.norm2(x)
        x = self.relu2(x)
        return x
