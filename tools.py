import torch

def train(model, train_loader, criterion, optimizer, device):
    model.train() 
    running_loss = 0.0
    
    for train_data in train_loader:
        ego_feat = torch.cat((train_data['ego_vec'], train_data['adj_veh_num'].unsqueeze(-1), train_data['adj_ped_num'].unsqueeze(-1)),dim=-1).float()
        adj_veh_feat = train_data['adj_veh_feat'].float()
        adj_ped_feat = train_data['adj_ped_feat'].float()
        gt = train_data['gt_prob'].float()
        
        ego_feat = ego_feat.to(device)
        adj_veh_feat = adj_veh_feat.to(device)
        adj_ped_feat = adj_ped_feat.to(device)
        gt = gt.to(device)

        optimizer.zero_grad() 
        outputs = model(ego_feat, adj_veh_feat, adj_ped_feat) 
        loss = criterion(outputs, gt) 
        loss.backward() 
        optimizer.step() 

        running_loss += loss.item() * ego_feat.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    #print(outputs)
    #print(gt)
    #print()
    return epoch_loss

def validate(model, val_loader, criterion, device):
    model.eval() 
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for valid_data in val_loader:
            ego_feat = torch.cat((valid_data['ego_vec'], valid_data['adj_veh_num'].unsqueeze(-1), valid_data['adj_ped_num'].unsqueeze(-1)),dim=-1).float()
            adj_veh_feat = valid_data['adj_veh_feat'].float()
            adj_ped_feat = valid_data['adj_ped_feat'].float()
            gt = valid_data['gt_prob'].float()
            
            ego_feat = ego_feat.to(device)
            adj_veh_feat = adj_veh_feat.to(device)
            adj_ped_feat = adj_ped_feat.to(device)
            gt = gt.to(device)

            outputs = model(ego_feat, adj_veh_feat, adj_ped_feat) 
            loss = criterion(outputs, gt) 

            running_loss += loss.item() * ego_feat.size(0)
            total_samples += ego_feat.size(0)

    avg_loss = running_loss / total_samples
    return avg_loss


def test(model, test_loader, criterion, device):
    model.eval()
    total_samples = 0

    with torch.no_grad():
        for test_data in test_loader:
            ego_feat = torch.cat((test_data['ego_vec'], test_data['adj_veh_num'].unsqueeze(-1), test_data['adj_ped_num'].unsqueeze(-1)),dim=-1).float()
            adj_veh_feat = test_data['adj_veh_feat'].float()
            adj_ped_feat = test_data['adj_ped_feat'].float()
            gt = test_data['gt_prob'].float()
            
            ego_feat = ego_feat.to(device)
            adj_veh_feat = adj_veh_feat.to(device)
            adj_ped_feat = adj_ped_feat.to(device)
            gt = gt.to(device)

            outputs = model(ego_feat, adj_veh_feat, adj_ped_feat)
            total_samples += ego_feat.size(0) 
            test_error = criterion(outputs, gt)
    
    return test_error