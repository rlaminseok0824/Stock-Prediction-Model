import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Multi-layer LSTM model for time series tasks
    Supports forecasting, classification, imputation, and anomaly detection
    """
    
    def __init__(self, configs):
        """
        Args:
            configs: Configuration object containing:
                - task_name: str ('long_term_forecast', 'short_term_forecast', 'classification', 'imputation', 'anomaly_detection')
                - seq_len: int, input sequence length
                - pred_len: int, prediction length (for forecasting)
                - enc_in: int, input feature dimension
                - hidden_size: int, LSTM hidden size
                - num_class: int, number of classes (for classification)  
                - dropout: float, dropout rate
        """
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.hidden_size = configs.hidden_size
        self.input_size = configs.enc_in
        
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
            
        # 3-layer LSTM
        self.lstm1 = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            dropout=configs.dropout if hasattr(configs, 'dropout') else 0.1
        )
        
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            dropout=configs.dropout if hasattr(configs, 'dropout') else 0.1
        )
        
        self.lstm3 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            dropout=configs.dropout if hasattr(configs, 'dropout') else 0.1
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.1)
        
        # Task-specific output layers
        if self.task_name == 'classification':
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.1),
                nn.Linear(self.hidden_size // 2, configs.num_class)
            )
        else:
            # For forecasting, imputation, anomaly detection
            self.output_projection = nn.Linear(self.hidden_size, self.input_size)
            
        # Additional projection for forecasting tasks
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.temporal_projection = nn.Linear(self.seq_len, self.pred_len)
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden states for LSTM layers"""
        h0_1 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c0_1 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        h0_2 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c0_2 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        h0_3 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c0_3 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        
        return (h0_1, c0_1), (h0_2, c0_2), (h0_3, c0_3)
    
    def encoder(self, x):
        """
        Encode input through 3-layer LSTM
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
        Returns:
            encoded output and final hidden state
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden states
        hidden_states = self.init_hidden(batch_size, device)
        
        # Pass through LSTM layers
        lstm1_out, hidden1 = self.lstm1(x, hidden_states[0])
        lstm1_out = self.dropout(lstm1_out)
        
        lstm2_out, hidden2 = self.lstm2(lstm1_out, hidden_states[1])
        lstm2_out = self.dropout(lstm2_out)
        
        lstm3_out, hidden3 = self.lstm3(lstm2_out, hidden_states[2])
        lstm3_out = self.dropout(lstm3_out)
        
        return lstm3_out, hidden3
    
    def forecast(self, x_enc):
        """Long-term or short-term forecasting"""
        encoded_out, _ = self.encoder(x_enc)  # [B, L, H]
        
        # Project hidden features back to input dimension
        output = self.output_projection(encoded_out)  # [B, L, D]
        
        # Temporal projection to prediction length
        if hasattr(self, 'temporal_projection'):
            output = output.transpose(1, 2)  # [B, D, L]
            output = self.temporal_projection(output)  # [B, D, pred_len]
            output = output.transpose(1, 2)  # [B, pred_len, D]
        
        return output
    
    def imputation(self, x_enc):
        """Missing value imputation"""
        encoded_out, _ = self.encoder(x_enc)  # [B, L, H]
        output = self.output_projection(encoded_out)  # [B, L, D]
        return output
    
    def anomaly_detection(self, x_enc):
        """Anomaly detection"""
        encoded_out, _ = self.encoder(x_enc)  # [B, L, H]
        output = self.output_projection(encoded_out)  # [B, L, D]
        return output
    
    def classification(self, x_enc):
        """Time series classification"""
        encoded_out, (h_n, c_n) = self.encoder(x_enc)  # [B, L, H]
        
        # Use the last hidden state for classification
        final_hidden = h_n.squeeze(0)  # [B, H]
        
        # Alternatively, you can use global average pooling:
        # final_hidden = encoded_out.mean(dim=1)  # [B, H]
        
        output = self.classifier(final_hidden)  # [B, num_classes]
        return output
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass
        Args:
            x_enc: Encoder input [batch_size, seq_len, input_size]
            x_mark_enc: Encoder timestamp features (optional)
            x_dec: Decoder input (optional, for some forecasting models)
            x_mark_dec: Decoder timestamp features (optional)
            mask: Mask for missing values (optional)
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out  # [B, pred_len, D]
        
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, num_classes]
        
        return None