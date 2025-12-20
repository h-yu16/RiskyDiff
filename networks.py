import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models  # https://github.com/pytorch/hub/issues/46
from transformers import BertModel, DistilBertModel, GPT2Model, AutoModel
import timm
from torch.hub import load_state_dict_from_url


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MLP(nn.Module):

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
                                      for _ in range(hparams['mlp_depth'] - 2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class PretrainedImageModel(torch.nn.Module):

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters."""
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ResNet(PretrainedImageModel): # Atually not limited to resnet. Need to make names consistent when loading models from DomainBed-v2.

    def __init__(self, input_shape, hparams, freeze_bn=False):
        super(ResNet, self).__init__()

        if hparams["arch"] == "resnet18":
            self.network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
            self.n_outputs = 512
        elif hparams["arch"] == "resnet50":
            self.network = torchvision.models.resnet50(pretrained=hparams["pretrained"])
            self.n_outputs = 2048
        elif hparams["arch"] == "vitb16":
            self.network = torchvision.models.vit_b_16(pretrained=hparams["pretrained"])
            self.n_outputs = 768
        elif hparams["arch"] == "densenet121":
            self.network = torchvision.models.densenet121(pretrained=hparams["pretrained"])
            self.n_outputs = 1024
        elif hparams["arch"] == "effnetb2":
            self.network = torchvision.models.efficientnet_b2(pretrained=hparams["pretrained"])
            self.n_outputs = 1408
        else:
            raise NotImplementedError

        assert input_shape[0] == 3

        # save memory
        if "resnet" in hparams["arch"]:
            del self.network.fc
            self.network.fc = Identity()
        elif "vit" in hparams["arch"]:
            if "heads" in dir(self.network):
                del self.network.heads
                self.network.heads = Identity()
        elif "densenet" in hparams["arch"] or "effnet" in hparams["arch"]:
            del self.network.classifier
            self.network.classifier = Identity()
        else:
            raise NotImplementedError

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['last_layer_dropout'])

        if freeze_bn:
            self.freeze_bn()
        else:
            assert hparams['last_layer_dropout'] == 0.

class MNIST_CNN(nn.Module):

    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        # x = F.normalize(x, dim=1)
        return x


class BertFeatureWrapper(torch.nn.Module):

    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.n_outputs = model.config.hidden_size
        classifier_dropout = (
            hparams['last_layer_dropout'] if hparams['last_layer_dropout'] != 0. else model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, x):
        kwargs = {
            'input_ids': x[:, :, 0],
            'attention_mask': x[:, :, 1]
        }
        if x.shape[-1] == 3:
            kwargs['token_type_ids'] = x[:, :, 2]
        output = self.model(**kwargs)
        if hasattr(output, 'pooler_output'):
            return self.dropout(output.pooler_output)
        else:
            return self.dropout(output.last_hidden_state[:, 0, :])


def replace_module_prefix(state_dict, prefix, replace_with=""):
    state_dict = {
        (key.replace(prefix, replace_with, 1)
         if key.startswith(prefix) else key): val
        for (key, val) in state_dict.items()
    }
    return state_dict


def get_torchvision_state_dict(url):
    model = load_state_dict_from_url(url)
    model_trunk = model["classy_state_dict"]["base_model"]["model"]["trunk"] if 'classy_state_dict' in model else model
    return replace_module_prefix(model_trunk, "_feature_blocks.")


def imagenet_resnet50_ssl(URL):
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Identity()
    model.load_state_dict(get_torchvision_state_dict(URL))
    model.fc.in_features = 2048
    model.n_outputs = 2048
    return model


def load_swag(URL):
    m = torchvision.models.vit_b_16(pretrained=False)
    m.heads = torch.nn.Identity()    
    state_dict = load_state_dict_from_url(URL)
    state_dict_new = {}
    for (key, val) in state_dict.items():
        if 'layer_' in key:
            key = key.replace('layer_', 'encoder_layer_', 1)
        if key == 'encoder.pos_embedding':
            val = val.permute((1, 0, 2))        
        state_dict_new[key] = val 
    m.load_state_dict(state_dict_new)
    m.n_outputs = 768
    return m


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given data type & input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError
    
def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)
