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
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, C, T)"""
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


# =====================================
# Temporal Attention (修正あり)
# =====================================
class TemporalAttention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """x: (B, T, H)"""
        scores = self.attn(x)  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)

        # 修正: 加重平均をとって時間軸 (T) を潰す
        # (B, T, H) * (B, T, 1) -> (B, T, H) -> sum over T -> (B, H)
        context = torch.sum(x * weights, dim=1)

        return context, weights


# =====================================
# Model (修正あり)
# =====================================
class CNN_GRU_Attention(nn.Module):

    def __init__(
        self,
        in_channels=8,
        conv_channels_1=16,
        conv_channels_2=32,
        gru_hidden_size=64,
        dropout_cnn=0.2,
        dropout_gru=0.3,
    ):
        super().__init__()

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels, conv_channels_1, kernel_size=15, padding=7
            ),
            nn.BatchNorm1d(conv_channels_1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_cnn),
            nn.Conv1d(
                conv_channels_1, conv_channels_2, kernel_size=7, padding=3
            ),
            nn.BatchNorm1d(conv_channels_2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_cnn),
        )

        # SE Block
        self.se = SEBlock(conv_channels_2)

        # GRU
        self.gru = nn.GRU(
            input_size=conv_channels_2,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.gru_dropout = nn.Dropout(dropout_gru)

        # Temporal Attention
        self.temporal_attention = TemporalAttention(gru_hidden_size)

        # 出力
        self.fc = nn.Linear(gru_hidden_size, 1)

    def forward(self, x):
        """x: (B, T, C)"""
        # CNN (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)

        # SE Block (B, C, T)
        x = self.se(x)

        # GRU (B, T, C)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = self.gru_dropout(out)

        # Temporal Attention (B, H) になる
        context, weights = self.temporal_attention(out)

        # 出力 (B, 1)
        out = self.fc(context)

        return out  


# 動作確認用
if __name__ == "__main__":
    model = CNN_GRU_Attention()
    dummy_input = torch.randn(32, 100, 8)  # (Batch=32, Time=100, Channel=8)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 期待値: torch.Size([32, 1])