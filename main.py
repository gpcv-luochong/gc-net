import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
import tensorboardX as tX
import os
import shutil
from network import *
from read_data import *
from torch.autograd import Variable
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

h=256
w=512
maxdisp=96 #gc_net.py also need to change  must be a multiple of 32...maybe can cancel the outpadding of deconv
batch=1
num_epochs = 200
save_per_epoch = 1
mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
device_ids = [0]

writer = tX.SummaryWriter(log_dir='log', comment='GCNet')
device = torch.device('cuda')
print(device)

#train
def main():
    train_transform = T.Compose([RandomCrop([h,w]), Normalize(mean, std), ToTensor()])
    train_dataset = KITTI2015('H:/lc\scene_flow', mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

    validate_transform = T.Compose([Normalize(mean, std), ToTensor(),Pad(384,1248)])
    validate_dataset = KITTI2015('H:/lc\scene_flow', mode='validate', transform=validate_transform)
    validate_loader = DataLoader(validate_dataset, batch_size=1, num_workers=1)

    step = 0
    best_error = 100.0

    model = GcNet(h,w,maxdisp)
    model = nn.DataParallel(model)
    criterion = SmoothL1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    for epoch in range(1,num_epochs + 1):
        model.train()
        step = train(model, train_loader, optimizer, criterion, step)
        adjust_lr(optimizer, epoch)

        if epoch % save_per_epoch == 0:
            model.eval()
            error = validate(model, validate_loader, epoch)
            best_error = save(model, optimizer, epoch, step, error, best_error)



def train(model, train_loader, optimizer, criterion, step):
    '''
    train one epoch
    '''
    for batch in train_loader:
        step += 1
        optimizer.zero_grad()

        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)

        mask = (target_disp > 0)
        mask = mask.detach_()

        disp = model(left_img, right_img)

        loss= criterion(disp[mask] , target_disp[mask])

        loss.backward()
        optimizer.step()

        # print(step)

        if step % 1 == 0:
            writer.add_scalar('loss', loss, step)
            print('step: {:05} | total loss: {:.5}'.format(step, loss.item()))

    return step

def validate(model, validate_loader, epoch):
    '''
    validate 40 image pairs
    '''
    num_batches = len(validate_loader)
    idx = np.random.randint(num_batches)

    avg_error = 0.0
    for i, batch in enumerate(validate_loader):
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)

        mask = target_disp.gt(0)
        mask = mask.detach_()
        # plt.figure()
        # plt.imshow(target_disp.cpu().numpy().transpose(1,2,0).squeeze(2))
        # plt.show()
        # plt.close()
        with torch.no_grad():
            disp = model(left_img, right_img)

        delta = torch.abs(disp[mask] - target_disp[mask])
        error_mat = (((delta >= 3.0) + (delta >= 0.05 * (target_disp[mask]))) == 2)
        error = torch.sum(error_mat).item() / torch.numel(disp[mask]) * 100
        # error = torch.sum(delta > 3.0) / float(h * w )

        avg_error += error
        if i == idx:
            left_save = left_img
            disp_save = disp

    avg_error = avg_error / num_batches
    print('epoch: {:03} | 3px-error: {:.5}%'.format(epoch, avg_error))
    writer.add_scalar('error/3px', avg_error, epoch)
    save_image(left_save[0], disp_save[0], epoch)

    return avg_error

def save_image(left_image, disp, epoch):
    for i in range(3):
        left_image[i] = left_image[i] * std[i] + mean[i]
    b, r = left_image[0], left_image[2]
    left_image[0] = r  # BGR --> RGB
    left_image[2] = b
    # left_image = torch.from_numpy(left_image.cpu().numpy()[::-1])

    disp_img = disp.detach().cpu().numpy()
    fig = plt.figure( figsize=(12.84, 3.84) )
    plt.axis('off')  # hide axis
    plt.imshow(disp_img)
    plt.colorbar()

    writer.add_figure('image/disp', fig, global_step=epoch)
    writer.add_image('image/left', left_image, global_step=epoch)

def adjust_lr(optimizer, epoch):
    if epoch == 200:
        lr = 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def save(model, optimizer, epoch, step, error, best_error):
    path = os.path.join('model', '{:03}.ckpt'.format(epoch))
    # torch.save(model.state_dict(), path)
    # model.save_state_dict(path)

    state = {}
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['error'] = error
    state['epoch'] = epoch
    state['step'] = step

    torch.save(state, path)
    print('save model at epoch{}'.format(epoch))

    if error < best_error:
        best_error = error
        best_path = os.path.join('model', 'best_model.ckpt'.format(epoch))
        shutil.copyfile(path, best_path)
        print('best model in epoch {}'.format(epoch))

    return best_error

class SmoothL1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, disp1,  target):
        loss1 = F.smooth_l1_loss(disp1, target)

        return loss1




if __name__=='__main__':
    main()
    writer.close()
