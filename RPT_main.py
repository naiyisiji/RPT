import torch
from torch.nn import KLDivLoss, MSELoss
from RPT import RPT
from survey_loader import loader
train_loader, val_loader, test_loader = loader('D:\\Nuscenes\\survey\\processed', 1, shuffle=False)
from tools import train, validate, test

ego_input_dim = 4
veh_input_dim = 5
ped_input_dim = 3
hidden_dim = 128
num_heads = 8
dropout = 0.1
adj_veh_hidden_layer = 1
adj_ped_hidden_layer = 1

num_epochs = 300
best_val_loss = float('inf')
device = torch.device('cuda')
model = RPT(ego_input_dim,
                      veh_input_dim,
                      ped_input_dim,
                      hidden_dim,
                      num_heads,
                      dropout).to(device)
criterion = MSELoss()
test_error = MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5, weight_decay=1e-6)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Model saved!")

model.load_state_dict(torch.load('best_model.pth')) 
test_accuracy = test(model, test_loader, test_error, device)
print(f"Test Accuracy: {test_accuracy:.4f}")