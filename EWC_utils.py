import torch
import torch.nn.functional as F
import numpy as np

class EWC:
    def __init__(self, model, env, n_samples=2048, mini_batch_size=64, importance=0.4, device="cuda"):
        self.model = model
        self.env = env
        self.n_samples = n_samples
        self.mini_batch_size = mini_batch_size
        self.importance = importance
        self.device = device if device != 'auto' else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params = {n: p.to(self.device) for n, p in self.model.policy.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrix = self._diag_fisher()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach().to(self.device)
        
    def _diag_fisher(self):
        _precision_matrix = {n: torch.zeros_like(p).to(self.device) for n, p in self.params.items()}
        
        self.model.policy.eval()
        observations = []
        obs = self.env.reset()
        for _ in range(self.n_samples):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, _ = self.env.step(action)
            observations.append(obs)
            if np.any(dones):  # If any of the environments are done
                obs = self.env.reset()

        for start in range(0, len(observations), self.mini_batch_size):
            end = start + self.mini_batch_size
            batch_obs = np.array(observations[start:end])
            batch_obs = torch.tensor(batch_obs, dtype=torch.float32).to(self.device)

            # Forward pass through the policy network
            features = self.model.policy.features_extractor(batch_obs)
            actor_data = self.model.policy.mlp_extractor.forward_actor(features)
            critic_data = self.model.policy.mlp_extractor.forward_critic(features)
            actor_logits = self.model.policy.action_net(actor_data)
            value_estimates = self.model.policy.value_net(critic_data)        

            target = torch.argmax(actor_logits, dim=1).long()
            actor_loss = F.cross_entropy(actor_logits, target)  # Use cross-entropy loss
            critic_loss = F.mse_loss(value_estimates, torch.zeros_like(value_estimates))  # Critic loss
            
            total_loss = actor_loss + critic_loss
            self.model.policy.zero_grad()  # Zero the gradients before backward pass
            total_loss.backward()

            for n, p in self.model.policy.named_parameters():
                if p.requires_grad and p.grad is not None:
                    _precision_matrix[n] += p.grad.data.clone().pow(2)

        self.model.policy.train()

        return _precision_matrix

    def penalty(self):
        loss = 0
        for n, p in self.model.policy.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrix[n] * (p - self._means[n]).pow(2)
                loss += _loss.sum()
        return loss
