import torch

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration of order 5 to compute the sign/orthogonal component of G.
    Optimized to run on the smaller dimension.
    """
    transposed = False
    if G.size(0) < G.size(1):
        G = G.t()
        transposed = True

    X = G.to(torch.float32)
    # Normalize
    X = X / (X.norm() + eps)
    
    # Polynomial coefficients for order 5
    a, b, c = 3.4445, -4.7750, 2.0315
    
    for _ in range(steps):
        # X shape: (M, N) with M >= N
        # XTX shape: (N, N) - smaller dimension
        XTX = torch.mm(X.t(), X)
        # Update: X = a*X + b*X*XTX + c*X*XTX^2
        X = a * X + torch.mm(X, b * XTX + c * torch.mm(XTX, XTX))
        
    if transposed:
        X = X.t()
    return X.type_as(G)

class Muon(torch.optim.Optimizer):
    """
    Muon: An optimizer for 2D weights in neural networks.
    Applies Newton-Schulz orthogonalization to weight updates.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                
                # Retrieve or initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                buf = state['momentum_buffer']
                
                # Momentum update
                buf.mul_(momentum).add_(grad)
                
                # Nesterov momentum option
                if nesterov:
                    g = grad.add(buf, alpha=momentum)
                else:
                    g = buf
                    
                # Apply order-5 Newton-Schulz orthogonalization
                g = zeropower_via_newtonschulz5(g)
                
                # Apply weight update
                p.data.add_(g, alpha=-lr)

