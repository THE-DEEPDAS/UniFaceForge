"""Evaluate a saved UniFaceForge checkpoint, including ROC AUC and EER."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.unifaceforge.model import UniFaceForge

def auc_eer(scores, labels):
    order=torch.argsort(scores,descending=True); y=labels[order]
    pos=(y==1).sum().item(); neg=(y==0).sum().item()
    tp=fp=0; points=[(0.,0.)]; best=(1.,1.)
    for v in y.tolist():
        if v==1: tp+=1
        else: fp+=1
        tpr=tp/pos if pos else 0.; fpr=fp/neg if neg else 0.
        points.append((fpr,tpr)); far=fpr; frr=1-tpr
        if abs(far-frr)<best[0]: best=(abs(far-frr),(far+frr)/2)
    area=sum((x2-x1)*(y1+y2)/2 for (x1,y1),(x2,y2) in zip(points,points[1:]))
    return area,best[1]

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument("data",type=Path); p.add_argument("checkpoint",type=Path); p.add_argument("--batch-size",type=int,default=8); a=p.parse_args()
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    ds=datasets.ImageFolder(a.data,transform=tfm); loader=DataLoader(ds,a.batch_size,num_workers=0)
    ck=torch.load(a.checkpoint,map_location="cpu"); m=UniFaceForge(False); m.load_state_dict(ck["model"]); m.eval()
    fake_index=ds.class_to_idx.get("fake")
    if fake_index is None: raise ValueError("dataset must contain a class directory named 'fake'")
    scores=[]; labels=[]
    for x,y in loader: scores.append(m(x).softmax(1)[:,fake_index]); labels.append((y==fake_index).long())
    scores=torch.cat(scores); labels=torch.cat(labels); auc,eer=auc_eer(scores,labels)
    print(json.dumps({"classes":ds.classes,"n":len(ds),"roc_auc":auc,"eer":eer},indent=2))

if __name__ == "__main__": main()
