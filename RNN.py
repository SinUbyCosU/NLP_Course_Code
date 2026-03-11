import torch
import torch.nn as nn

# 1. BARE-BONES DATA: Sequences of 3 numbers, predicting the 4th
# X shapes into (100 batches, 3 time steps, 1 feature)
X = torch.tensor([[[i], [i+1], [i+2]] for i in range(100)], dtype=torch.float32)
y = torch.tensor([[i+3] for i in range(100)], dtype=torch.float32)

# 2. BARE-BONES MODEL
class MinimalRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=10, batch_first=True)
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        out, _ = self.rnn(x)          # Run sequence through RNN
        return self.linear(out[:, -1, :]) # Only output the final step's prediction

model = MinimalRNN()

# 3. TRAINING LOOP
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

for epoch in range(150):
    optimizer.zero_grad()       # Clear old math
    loss = loss_fn(model(X), y) # Make guesses and calculate error
    loss.backward()             # Figure out how to fix error
    optimizer.step()            # Update the model

# 4. TEST IT OUT
test_seq = torch.tensor([[[100], [101], [102]]], dtype=torch.float32)
print(f"Prediction for [100, 101, 102]: {model(test_seq).item():.2f}")