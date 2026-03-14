import torch
import torch.nn as nn


class CrowdNavNet(nn.Module):

    def __init__(self,
                 spatial_dim,
                 temporal_dim,
                 goal_dim,
                 human_pref_dim=3,
                 n_robot_actions=5,
                 n_human_actions=3):
        super().__init__()

        # =====================================
        # SPATIAL 3-LAYER RNN
        # 128 -> 64
        # =====================================
        self.spatial_net = nn.GRU(
            input_size=spatial_dim,
            hidden_size=64,
            num_layers=2,          # stacked layers
            batch_first=True
        )

        # =====================================
        # TEMPORAL 3-LAYER RNN
        # 128 -> 64
        # =====================================
        self.temporal_net = nn.GRU(
            input_size=temporal_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        # =====================================
        # GOAL → 64 (single FC)
        # =====================================
        self.goal_fc = nn.Linear(goal_dim, 64)

        # =====================================
        # OUTPUT DIMENSION (paper)
        # =====================================
        self.output_dim = 64 + 64 + 64 + human_pref_dim   # 195

    # =====================================
    # FORWARD
    # =====================================
    def forward(self, spatial, temporal, goal, human_pref):

        # spatial RNN
        _, h_spatial = self.spatial_net(spatial)
        spatial_feat = h_spatial[-1]   # last layer hidden

        # temporal RNN
        _, h_temporal = self.temporal_net(temporal)
        temporal_feat = h_temporal[-1]

        # goal
        goal_feat = self.goal_fc(goal)

        # concat DIRECT (no fusion MLP)
        fused = torch.cat(
            [spatial_feat, temporal_feat, goal_feat, human_pref],
            dim=1
        )

        return fused