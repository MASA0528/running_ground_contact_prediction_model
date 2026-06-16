import torch
import torch.nn as nn


# =====================================
# SE Block
# =====================================
class SEBlock(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.LeakyReLU(0.1),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _ = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1)

        return x * y


# =====================================
# Temporal Attention
# =====================================
class TemporalAttention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: (B, T, H)
        """

        scores = self.attn(x)

        weights = torch.softmax(
            scores,
            dim=1
        )

        attended = x * weights

        return attended, weights


# =====================================
# CNN + GRU + Attention
# =====================================
class CNN_GRU_Attention(nn.Module):

    def __init__(
        self,
        in_channels=12,
        conv_channels_1=16,
        conv_channels_2=32,
        conv_channels_3=64,
        gru_hidden_size=64,
        fc_hidden_size=32,
        dropout_cnn1=0.13000844385332877,
        dropout_cnn2=0.2450249542948521,
        dropout_cnn3=0.333294632167325,
        dropout_gru=0.2917753403062322,
        dropout_fc=0.32014556438414565,
        gn_groups_1=4,
        gn_groups_2=8,
        gn_groups_3=8
    ):
        super().__init__()

        # =================================
        # Input LayerNorm
        # =================================
        self.input_ln = nn.LayerNorm(
            in_channels
            )

        # =================================
        # CNN Block 1
        # =================================
        self.conv1 = nn.Sequential(

            nn.Conv1d(
                in_channels=in_channels,
                out_channels=conv_channels_1,
                kernel_size=21,
                padding=10,
                stride=1
            ),

            nn.GroupNorm(
                num_groups=gn_groups_1,
                num_channels=conv_channels_1
            ),

            nn.LeakyReLU(0.2),

            nn.MaxPool1d(
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.Dropout1d(
                dropout_cnn1
            )
        )

        # =================================
        # CNN Block 2
        # =================================
        self.conv2 = nn.Sequential(

            nn.Conv1d(
                conv_channels_1,
                conv_channels_2,
                kernel_size=15,
                padding=7,
                stride=1
            ),

            nn.GroupNorm(
                num_groups=gn_groups_2,
                num_channels=conv_channels_2
            ),

            nn.LeakyReLU(0.2),

            nn.MaxPool1d(
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.Dropout1d(
                dropout_cnn2
            )
        )

        # =================================
        # CNN Block 3
        # =================================
        self.conv3 = nn.Sequential(

            nn.Conv1d(
                conv_channels_2,
                conv_channels_3,
                kernel_size=11,
                padding=5,
                stride=1
            ),

            nn.GroupNorm(
                num_groups=gn_groups_3,
                num_channels=conv_channels_3
            ),

            nn.LeakyReLU(0.2),

            nn.Dropout1d(
                dropout_cnn3
            )
        )

        # =================================
        # SE
        # =================================
        self.se = SEBlock(
            conv_channels_3
        )

        # =================================
        # GRU
        # =================================
        self.gru = nn.GRU(

            input_size=conv_channels_3,

            hidden_size=gru_hidden_size,

            num_layers=1,

            batch_first=True,

            bidirectional=False
        )

        self.gru_dropout = nn.Dropout(
            dropout_gru
        )

        # =================================
        # Attention
        # =================================
        self.temporal_attention = TemporalAttention(
            gru_hidden_size
        )

        # =================================
        # FC
        # =================================
        self.fc = nn.Sequential(

            nn.Linear(
                gru_hidden_size,
                fc_hidden_size
            ),


            nn.LeakyReLU(0.2),

            nn.Dropout(
                dropout_fc
            ),

            nn.Linear(
                fc_hidden_size,
                1
            )
        )

    def forward(self, x):
        """
        x: (B, T, C)
        """

        # =================================
        # Input LayerNorm
        # =================================
        x = self.input_ln(x)

        # =================================
        # (B,T,C) → (B,C,T)
        # =================================
        x = x.permute(0, 2, 1)

        # =================================
        # CNN
        # =================================
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        # =================================
        # SE
        # =================================
        x = self.se(x)

        # =================================
        # (B,C,T) → (B,T,C)
        # =================================
        x = x.permute(0, 2, 1)

        # =================================
        # GRU
        # =================================
        out, _ = self.gru(x)

        out = self.gru_dropout(out)

        # =================================
        # Attention
        # =================================
        out, weights = self.temporal_attention(out)

        # =================================
        # FC
        # =================================
        out = self.fc(out)

        # (B,T,1)
        return out
    
