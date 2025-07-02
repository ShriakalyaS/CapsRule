# Define Capsule Network
class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_capsules):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn(1, input_dim, num_capsules, output_dim))

    def forward(self, x):
        u_hat = torch.matmul(x.unsqueeze(2), self.W)
        b_ij = torch.zeros(*u_hat.size()[:-1], device=x.device)
        for _ in range(3):  # Routing iterations
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)
            v_j = self.squash(s_j)
            b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(-1)
        return v_j

    def squash(self, s):
        norm = torch.norm(s, dim=-1, keepdim=True)
        return (norm**2 / (1 + norm**2)) * s / (norm + 1e-8)

class FFCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(FFCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.caps = CapsuleLayer(hidden_dim, 8, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.caps(x)
        out = x.norm(dim=-1)
        return out

# Initialize model
input_dim = X_train_tensor.shape[1]
model = FFCN(input_dim=input_dim, hidden_dim=64, num_classes=len(le.classes_))
