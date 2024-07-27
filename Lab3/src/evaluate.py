import torch
from utils import dice_score

def evaluate(model, data):
    # implement the evaluation function here
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        result=0
        for sample in data:
            image, mask = sample["image"].to(device), sample["mask"].to(device)
            pred = model(image.to(dtype=torch.float32))
            ######## calculate dice score
            result += dice_score(pred,mask)
        score= result / len(data)
    return    score
    #assert False, "Not implemented yet!"