class MOE(nn.Module):
    def __init__(self, num_experts, context_length, cfg):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(cfg["emb_dim"], num_experts)
        self.experts = nn.ModuleList([swigluffn(d=cfg["emb_dim"], dff=cfg["dff"]) for _ in range(num_experts)])
        self.noize_w=nn.Linear(cfg["emb_dim"],num_experts)
        self.dropout=nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        gate_logits = self.gate(x_flat)
        gate_logits=self.dropout(gate_logits)
        noize=torch.rand_like(gate_logits)*nn.functional.softplus(self.noize_w(x)).view(-1,self.num_experts)
        weights, indices = (gate_logits+noize).topk(k=2, dim=-1)
        weights = nn.functional.softmax(weights, dim=-1)
        final_output = torch.zeros_like(x_flat,device=x.device)
        for expert_idx in range(self.num_experts):
            expert_mask = (indices == expert_idx).any(dim=-1)
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)
                token_weights = torch.zeros(expert_mask.sum(), device=x.device)
                mask_pos1 = indices[expert_mask, 0] == expert_idx
                mask_pos2 = indices[expert_mask, 1] == expert_idx
                token_weights[mask_pos1] = weights[expert_mask, 0][mask_pos1]
                token_weights[mask_pos2] = weights[expert_mask, 1][mask_pos2]
                final_output[expert_mask] += expert_output * token_weights.unsqueeze(-1)
        if self.training:
          probs=gate_logits.softmax(dim=-1)
          importance = probs.mean(0)
          load = torch.zeros(self.num_experts, device=x.device)
          for i in range(self.num_experts):
            load[i] = (indices == i).any(dim=-1).float().mean()
          load = load.detach() + 1e-9
          importance = importance.detach() + 1e-9
          load_balance_loss = self.num_experts * torch.sum(importance * load)
          return final_output.view(batch_size, seq_len, d_model), load_balance_loss
        else:
          return final_output.view(batch_size, seq_len, d_model)
