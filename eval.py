import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, valloader, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    loss = 0

    for i_val, samples_val in tqdm(enumerate(valloader), total=len(valloader), ncols=80, leave=False):
        
        with torch.no_grad():
            images_val = samples_val['image'].unsqueeze[0].cuda(non_blocking=True)
            labels_val = samples_val['label'].unsqueeze[0].cuda(non_blocking=True)
    
            outputs_val = net(images_val)[0]
        
            loss += dice_coeff(outputs_val, labels_val).item()

    return loss / (i_val + 1)
    
