import torch
import torch.nn as nn
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_dim=196, hidden_dim=128, n_actions=5, n_objectives=4, debug=False):
        super().__init__()
        self.n_actions = n_actions
        self.n_objectives = n_objectives
        self.debug = debug

        # =============================
        # TRUNK CONDIVISO
        # =============================
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(hidden_dim),
        )

        # =============================
        # VALUE STREAM: V(s) → [B, K]
        # Stima il valore scalare per obiettivo dello stato
        # Stream separato con MLP dedicato per evitare che
        # value domini sull'advantage nelle prime fasi
        # =============================
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim // 2, n_objectives)
        )

        # =============================
        # ADVANTAGE STREAM: A(s,a) → [B, A*K]
        # Stima il vantaggio per ogni azione e obiettivo
        # Stream separato con MLP dedicato per avere
        # gradiente indipendente dal value stream
        # =============================
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim // 2, n_actions * n_objectives)
        )

        # =============================
        # INIZIALIZZAZIONE PESI
        # Inizializza l'ultimo layer dell'advantage vicino a zero
        # così all'inizio Q ≈ V(s) per tutte le azioni → evita
        # collasso immediato durante le prime iterazioni
        # =============================
        nn.init.uniform_(self.advantage_stream[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.advantage_stream[-1].bias)

    def forward(self, state):
        x = self.trunk(state)

        # Value: [B, K] → [B, 1, K]
        value = self.value_stream(x).unsqueeze(1)

        # Advantage: [B, A*K] → [B, A, K]
        advantage = self.advantage_stream(x)
        advantage = advantage.view(-1, self.n_actions, self.n_objectives)

        # Dueling aggregation: sottrai media per stabilizzare
        # Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))

        if self.debug:
            action_spread = (q.max(dim=1).values - q.min(dim=1).values).mean().item()
            print(f"[DQN DEBUG] action_spread={action_spread:.6f} | "
                  f"value_mean={value.mean().item():.4f} | "
                  f"adv_std={advantage.std().item():.6f}")

        assert q.shape[-1] == self.n_objectives, f"Output {q.shape}, expected {self.n_objectives}"
        return q


# ======================
# REPLAY MEMORY
# ======================
class ReplayMemory:
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return iter(self.memory)
