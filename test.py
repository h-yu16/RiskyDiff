import torch
import argparse
import misc
import os
from collections import defaultdict as dd
import shutil
from datastuff import get_dataset_class, get_files_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--category', type=str, default="")
    parser.add_argument('--model_arch', type=str, default="")
    parser.add_argument('--datadir', type=str, default="")
    parser.add_argument('--targetdir', type=str, default="")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = get_dataset_class(args.dataset)()

    model = misc.load_model(dataset, args.model_arch)
    model.to(device)
    model.eval()
    
    names = []
    labels = []
    path_list = os.listdir(args.datadir)
    path_list.sort()
    for idx, path in enumerate(path_list):
        names.append(os.path.join(args.datadir, path))
        labels.append(10000)
    testloader = get_files_dataloader(names, labels, phase="test", batch_size=1024, img_tr=dataset.transform)
        
    predictions, confs = misc.get_model_predictions(model, testloader, device)
    assert len(predictions) == len(confs)
    assert len(predictions) == len(path_list)
    results = [(filename, dataset.label2class[pred], conf) for filename, pred, conf in zip(path_list, predictions, confs)]
    for item in results:
        if item[1] != args.category and args.targetdir != "":
            os.makedirs(os.path.join(args.targetdir, args.category), exist_ok=True)
            shutil.copy(os.path.join(args.datadir, item[0]), os.path.join(args.targetdir, args.category))
        print(item)
    cate_list = set([item[1] for item in results]) 
    cnt_dict = dd(int)
    for item in results:
        cnt_dict[item[1]] += 1
    cnt_dict = sorted(cnt_dict.items(), key=lambda x: x[1], reverse=True)
    for item in cnt_dict:
        print("%s: %.3f" % (item[0], item[1]/len(results)))
