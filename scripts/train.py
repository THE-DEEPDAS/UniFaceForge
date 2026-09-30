"""Train/evaluate the reproducible reference model.

Expected layout (ImageFolder-compatible):
data/
  real/ ...
  fake/ ...

The script deliberately records the seed, split sizes and checkpoint path so
reported metrics can be audited. It does not silently create the paper's
balanced 1,000-image diagnostic subset.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from src.unifaceforge.model import UniFaceForge

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

@torch.no_grad()
def evaluate(model, loader, device, positive_index):
    model.eval(); tp=tn=fp=fn=0
    for x, y in loader:
        p=(model(x.to(device)).argmax(1).cpu()==positive_index); y=(y==positive_index)
        tp += int(((p==1)&(y==1)).sum()); tn += int(((p==0)&(y==0)).sum())
        fp += int(((p==1)&(y==0)).sum()); fn += int(((p==0)&(y==1)).sum())
    n=tp+tn+fp+fn; precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0
    return {"n":n,"accuracy":(tp+tn)/n,"precision":precision,"recall":recall,
            "f1":2*precision*recall/(precision+recall) if precision+recall else 0,
            "confusion_matrix":{"tn":tn,"fp":fp,"fn":fn,"tp":tp}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("data", type=Path); ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",type=Path,default=Path("runs/unifaceforge.pt")); args=ap.parse_args()
    seed_everything(args.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    ds=datasets.ImageFolder(args.data, transform=tfm)
    if len(ds.classes)!=2: raise ValueError(f"expected exactly two classes, found {ds.classes}")
    n=len(ds); a=int(.70*n); b=int(.15*n); train,val,test=random_split(ds,[a,b,n-a-b],generator=torch.Generator().manual_seed(args.seed))
    loaders=[DataLoader(z,args.batch_size,shuffle=i==0,num_workers=0) for i,z in enumerate((train,val,test))]
    model=UniFaceForge().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4); loss=nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        model.train()
        for x,y in loaders[0]: opt.zero_grad(); loss(model(x.to(device)),y.to(device)).backward(); opt.step()
    # Keep training labels unchanged, but make the audit convention explicit:
    # label 1 is forged in the saved evaluation split.
    fake_index=ds.class_to_idx.get("fake")
    if fake_index is None: raise ValueError("dataset must contain a class directory named 'fake'")
    metrics=evaluate(model,loaders[2],device,fake_index); args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"classes":ds.classes,"seed":args.seed,"metrics":metrics},args.out)
    print(json.dumps({"device":str(device),"classes":ds.classes,"split_sizes":[len(train),len(val),len(test)],"metrics":metrics},indent=2))

if __name__ == "__main__": main()
