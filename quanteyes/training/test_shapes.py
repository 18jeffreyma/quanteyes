import torch
import torch.nn as nn


# Define your Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        encoder_outputs, (hidden, cell) = self.lstm(x)
        return encoder_outputs, hidden, cell


# Define your Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        print(f"decoder input shape: {x.shape}")
        x = x.unsqueeze(1)  # Add a time dimension (sequence length of 1)
        embedded = self.embedding(x)
        print(f"decoder embedded shape: {embedded.shape}")
        decoder_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.fc(decoder_output.squeeze(1))
        return output, hidden, cell


# Initialize your encoder and decoder
input_size = 100  # Vocabulary size of input language
output_size = 120  # Vocabulary size of output language
hidden_size = 256
num_layers = 2

encoder = Encoder(64, hidden_size, num_layers)
decoder = Decoder(output_size, hidden_size, num_layers)

batch_size = 32
input_sequence_length = 10
output_sequence_length = 5

encoder_input = torch.randint(0, input_size, (batch_size, input_sequence_length, 64))
decoder_input = torch.randint(0, output_size, (batch_size, output_sequence_length, 64))
encoder_outputs, encoder_hidden, encoder_cell = encoder(encoder_input)
decoder_output, _, _ = decoder(decoder_input[:, 0], encoder_hidden, encoder_cell)
