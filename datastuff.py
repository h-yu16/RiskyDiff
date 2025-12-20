import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from os.path import join
import pandas as pd
import json

IMAGENET_LABELS_FILE = "/home/hanyu/dataset/imagenet-simple-labels.json"

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return globals()[dataset_name]



class ImageNet:
    def __init__(self):
        self.num_classes = 1000
        self.input_shape = (3, 224, 224)
        with open(IMAGENET_LABELS_FILE, "r")as f:
            categories = json.load(f)
        self.categories = [category.replace(" ", "_") for category in categories]
        self.class2label = {category:idx for idx, category in enumerate(self.categories)}
        self.label2class = {v: k for k,v in self.class2label.items()}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert('RGB')),  # Ensure 3 channels (RGB)
            transforms.ToTensor(),        
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.decode_tr = get_default_transformer(decode=True)

def _dataset_info(txt_file):
    with open(txt_file, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.strip().split(' ')
        file_names.append(' '.join(row[:-1]))
        labels.append(int(row[-1]))

    return file_names, labels

class StandardDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None):
        self.names = names
        self.labels = labels

        self.N = len(self.names)
        self._image_transformer = img_transformer
    
    def get_image(self, index):
        # img = Image.open(self.names[index]).convert('RGB')
        img = Image.open(self.names[index])
        return self._image_transformer(img)
        
    def __getitem__(self, index):
        img = self.get_image(index)
        return img, int(self.labels[index])

    def __len__(self):
        return len(self.names)


def get_default_transformer(decode=False): # hard-coded
    if not decode:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_fix_dataloader(txtdir, dataset, domain, phase, batch_size, num_workers=8, img_tr=None):
    assert phase in ["val", "test"]
    names, labels = _dataset_info(join(txtdir, dataset, "%s_%s.txt"%(domain, phase)))
    if img_tr is None:
        img_tr = get_default_transformer()
    curDataset = StandardDataset(names, labels, img_tr)
    loader = data.DataLoader(dataset=curDataset, batch_size=batch_size, num_workers=num_workers)
    return loader

def get_files_dataloader(names, labels, phase, batch_size, num_workers=8, img_tr=None):
    assert phase in ["val", "test"]
    if img_tr is None:
        img_tr = get_default_transformer()
    curDataset = StandardDataset(names, labels, img_tr)
    loader = data.DataLoader(dataset=curDataset, batch_size=batch_size, num_workers=num_workers)
    return loader


def get_dataloaders(args, categories, tr=None, batch_size=1024):
    # set up dataloaders
    valloaders = [get_fix_dataloader(args.txtdir, args.dataset, domain, "val", batch_size, img_tr=tr) for domain in categories]
    testloaders = [get_fix_dataloader(args.txtdir, args.dataset, domain, "test", batch_size, img_tr=tr) for domain in categories]        
    
    for index, domain in enumerate(categories):
        print("Val %s size: %d" % (domain, len(valloaders[index].dataset)))
    for index, domain in enumerate(categories):
        print("Test %s size: %d" % (domain, len(testloaders[index].dataset)))
    return valloaders, testloaders


def get_test_dataloaders(args, hparams, categories, tr=None):
    testloaders = [get_fix_dataloader(args.txtdir, args.dataset, domain, "test", hparams["batch_size"], img_tr=tr) for domain in categories]        

    for index, domain in enumerate(categories):
        print("Test %s size: %d" % (domain, len(testloaders[index].dataset)))
    return testloaders