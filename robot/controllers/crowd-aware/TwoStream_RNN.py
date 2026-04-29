import torch
import torch.nn as nn


class CrowdNavNet(nn.Module):
    def __init__(self,
                 spatial_dim,
                 temporal_dim,
                 goal_dim,
                 human_pref_dim=4,
                 debug=False):
        super().__init__()

        self.debug = debug

        # =============================
        # SPATIAL GRU (3 layers)
        # =============================
        self.spatial_gru = nn.GRU(
            input_size=spatial_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # =============================
        # TEMPORAL GRU (3 layers)
        # =============================
        self.temporal_gru = nn.GRU(
            input_size=temporal_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # =============================
        # GOAL
        # =============================
        self.goal_fc = nn.Linear(goal_dim, 64)

        self.activation = nn.LeakyReLU()

        # =============================
        # OUTPUT DIM
        # =============================
        self.output_dim = 64 + 64 + 64 + human_pref_dim  # = 195

    def forward(self, spatial, temporal, goal, human_pref):

        # =============================
        # SPATIAL
        # =============================
        _, h_spatial = self.spatial_gru(spatial)
        spatial_feat = h_spatial[-1]   # ultimo layer → [B, 64]

        # =============================
        # TEMPORAL
        # =============================
        _, h_temporal = self.temporal_gru(temporal)
        temporal_feat = h_temporal[-1]  # [B, 64]

        # =============================
        # GOAL
        # =============================
        goal_feat = self.activation(self.goal_fc(goal))  # [B, 64]

        # =============================
        # FUSION
        # =============================
        fused = torch.cat(
            [spatial_feat, temporal_feat, goal_feat, human_pref],
            dim=1
        )

        return fused
