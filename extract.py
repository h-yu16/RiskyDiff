import torch
import argparse
from datastuff import get_dataloaders, get_dataset_class
import misc
import open_clip
import os
import pickle
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ImageNet")
    parser.add_argument('--model_arch', type=str, default="")
    parser.add_argument('--pkldir', type=str, default="pkl")
    parser.add_argument('--txtdir', type=str, default="/home/hanyu/dataset/txtlist-eval")
    parser.add_argument('--fe_arch', type=str, default="ViT-H-14")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = get_dataset_class(args.dataset)()

    model = misc.load_model(dataset, args.model_arch)
    model.to(device)
    model.eval()

    print("Start to load CLIP!")
    model_fe, _, preprocess = open_clip.create_model_and_transforms(args.fe_arch, pretrained='laion2b_s32b_b79k', device=device)
    model_fe.eval()
    model_fe.to(device)
    
    val_loaders, test_loaders = get_dataloaders(args, dataset.categories, tr=dataset.transform)
    dataloaders = dict()
    dataloaders["val"] = val_loaders
    dataloaders["test"] = test_loaders
    
        
    results = dict()
    features_dict = dict()
    for phase in ["val", "test"]:
        results[phase] = dict()
        features_dict[phase] = dict()
        for category, dataloader in zip(dataset.categories, dataloaders[phase]):
            # extract features
            features, _ = misc.get_features(args, model_fe, preprocess, category, phase, device)
            features_dict[phase][category] = features
            # calculate predictions and error
            results[phase][category] = dict()
            probs, losses, labels, acc, correct = misc.get_model_results(model, dataloader, device)
            results[phase][category]["labels"] = labels
            results[phase][category]["losses"] = losses
            preds = np.argmax(probs, axis=-1)
            results[phase][category]["error"] = (labels!=preds).astype(np.int32)
            results[phase][category]["acc"] = acc
            results[phase][category]["filepaths"] = dataloader.dataset.names
            print("True accuracy for %s %s: %.4f"% (phase, category, acc))

            
    os.makedirs(args.pkldir, exist_ok=True)
    with open(os.path.join(args.pkldir, f"{args.dataset}-{args.model_arch}.pkl"), "wb") as f:
        pickle.dump(results, f)
    with open(os.path.join(args.pkldir, f"{args.dataset}-features.pkl"), "wb") as f:
        pickle.dump(features_dict, f)        
            
    # category merge
    results_all = dict()
    features_all = dict()
    for phase in ["val", "test"]:
        results_all[phase] = dict()
        for key in ["labels", "losses", "error"]:
            results_all[phase][key] = np.concatenate([results[phase][category][key] for category in dataset.categories])
            print("%s:"%key, results_all[phase][key].shape)
        features_all[phase] = np.concatenate([features_dict[phase][category] for category in dataset.categories])
    
    with open(os.path.join(args.pkldir, f"{args.dataset}-{args.model_arch}-merge.pkl"), "wb") as f:
        pickle.dump(results_all, f)
    
    with open(os.path.join(args.pkldir, f"{args.dataset}-features-merge.pkl"), "wb") as f:
        pickle.dump(features_all, f)        
        
