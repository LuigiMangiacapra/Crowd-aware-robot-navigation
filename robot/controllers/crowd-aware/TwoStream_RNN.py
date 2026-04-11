import torch
import torch.nn as nn


class CrowdNavNet(nn.Module):

    def __init__(self,
                 spatial_dim,
                 temporal_dim,
                 goal_dim,
                 human_pref_dim=4,
                 n_robot_actions=5,
                 n_human_actions=4):
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
        # GOAL → 64 (single Fully Connected)
        # =====================================
        self.goal_fc = nn.Linear(goal_dim, 64)

        # =====================================
        # LAYER NORMALIZATION  ← aggiunto
        # =====================================
        self.ln_spatial = nn.LayerNorm(64)
        self.ln_temporal = nn.LayerNorm(64)
        self.ln_goal = nn.LayerNorm(64)

        # =====================================
        # OUTPUT DIMENSION
        # =====================================
        self.output_dim = 64 + 64 + 64 + human_pref_dim

    # =====================================
    # FORWARD
    # =====================================
    def forward(self, spatial, temporal, goal, human_pref):
        _, h_spatial = self.spatial_net(spatial)
        spatial_feat = self.ln_spatial(h_spatial[-1])

        _, h_temporal = self.temporal_net(temporal)
        temporal_feat = self.ln_temporal(h_temporal[-1])

        goal_feat = self.ln_goal(self.goal_fc(goal))

        fused = torch.cat(
            [spatial_feat, temporal_feat, goal_feat, human_pref],
            dim=1
        )
        return fused

    # h_spatial = None
    # h_temporal = None

    # def forward(self, spatial, temporal, goal, human_pref, h_spatial=None, h_temporal=None):
    #     out_s, h_spatial_new = self.spatial_net(spatial,  h_spatial)
    #     out_t, h_temporal_new = self.temporal_net(temporal, h_temporal)

    #     spatial_feat = h_spatial_new[-1]
    #     temporal_feat = h_temporal_new[-1]

    #     goal_feat = self.goal_fc(goal)
    #     fused = torch.cat([spatial_feat, temporal_feat, goal_feat, human_pref], dim=1)

    #     return fused, h_spatial_new, h_temporal_new
