import torch
from torch.nn import KLDivLoss, MSELoss
from baseline import BaseLine
from survey_loader import loader
train_loader, val_loader, test_loader = loader('D:\\Nuscenes\\survey\\processed', 1, shuffle=False)
from tools import train, validate, test

ego_input_dim = 4
veh_input_dim = 5
ped_input_dim = 3
hidden_dim = 256

num_epochs = 300
best_val_loss = float('inf')
device = torch.device('cuda')
model = BaseLine(ego_input_dim,
                         veh_input_dim,
                         ped_input_dim,
                         hidden_dim).to(device)
criterion = MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5, weight_decay=1e-6)

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    if epoch % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_baseline.pth')
        print(f"Validation Loss: {val_loss:.4f}")
        print("Model saved!")

model.load_state_dict(torch.load('best_baseline.pth')) 
test_error = test(model, test_loader, device)
print(f"Test Accuracy: {test_error:.4f}")