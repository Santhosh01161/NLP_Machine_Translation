import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request
from indicnlp.tokenize import indic_tokenize
import os

app = Flask(__name__)

# ==========================================
# 1. Configuration & Path Setup
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
TEMPLATE_DIR = os.path.join(CURRENT_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)

def get_parent_path(filename):
    return os.path.join(PARENT_DIR, filename)

# Special Tokens
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<pad>', '<sos>', '<eos>'] # Removed <unk> so we can see it!

# ==========================================
# 2. Tokenizer
# ==========================================
class CustomTokenizer:
    def __init__(self, vocab_filename, language='en'):
        self.language = language
        self.word2idx = {}
        self.idx2word = {}
        
        file_path = get_parent_path(vocab_filename)
        try:
            # Force load on CPU
            self.word2idx = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
            
            if hasattr(self.word2idx, 'word2idx'):
                self.word2idx = self.word2idx.word2idx
            
            self.idx2word = {v: k for k, v in self.word2idx.items()}
            print(f"DEBUG: Loaded {len(self.word2idx)} tokens for {language}.")
            
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load {file_path}. Error: {e}")

    def encode(self, text):
        tokens = self.tokenize(text)
        # Debug: Print tokens found
        indices = [SOS_IDX] + [self.word2idx.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
        return indices

    def tokenize(self, text):
        if self.language == 'ta':
            return indic_tokenize.trivial_tokenize(text)
        else:
            return text.lower().split()

    def decode(self, indices):
        tokens = []
        for idx in indices:
            if isinstance(idx, torch.Tensor): idx = idx.item()
            word = self.idx2word.get(idx, '<unk>')
            # We now allow <unk> to be shown
            if word not in special_symbols:
                tokens.append(word)
        return ' '.join(tokens)

# Load Vocab
src_tokenizer = CustomTokenizer('en_vocab.pth', language='en')
trg_tokenizer = CustomTokenizer('ta_vocab.pth', language='ta')

# ==========================================
# 3. Model Architecture (ADDITIVE ATTENTION)
# ==========================================
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.Wa = nn.Linear(self.head_dim, self.head_dim)
        self.Ua = nn.Linear(self.head_dim, self.head_dim)
        self.V = nn.Linear(self.head_dim, 1)
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.fc_k(key).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.fc_v(value).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        Q_expanded = self.Wa(Q).unsqueeze(-2)  
        K_expanded = self.Ua(K).unsqueeze(-3)
        energy = self.V(torch.tanh(Q_expanded + K_expanded)).squeeze(-1)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hid_dim)
        return self.fc_o(x), attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hid_dim, pf_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pf_dim, hid_dim)
        )
    def forward(self, x):
        return self.layers(x)

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=500):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=500):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        return trg_pad_mask & trg_sub_mask
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

# ==========================================
# 4. Initialize Model (Additive Configuration)
# ==========================================
INPUT_DIM = len(src_tokenizer.word2idx)
OUTPUT_DIM = len(trg_tokenizer.word2idx)
HID_DIM = 128
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 4  # Adjusted for Additive
DEC_HEADS = 4
ENC_PF_DIM = 256
DEC_PF_DIM = 256
DROPOUT = 0.1

enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, DROPOUT, DEVICE)
dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DROPOUT, DEVICE)
model = Seq2Seq(enc, dec, PAD_IDX, PAD_IDX, DEVICE).to(DEVICE)

try:
    path = get_parent_path("en-ta-transformer-additive.pt")
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False))
    print(f"SUCCESS: Loaded Additive model from {path}")
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODEL: {e}")

# ==========================================
# 5. Translation Logic (With Debugging)
# ==========================================
def translate_sentence(sentence, max_len=50):
    model.eval()
    
    # Debug: Check tokens
    tokens = src_tokenizer.encode(sentence)
    print(f"\nDEBUG: Input Tokens: {tokens}")
    
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indices = [SOS_IDX]
    
    # Debug: Track predicted tokens
    predicted_tokens = []
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(DEVICE)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:, -1].item()
        
        # Debug print every step
        word = trg_tokenizer.idx2word.get(pred_token, '???')
        # print(f"Step {i}: ID={pred_token} Word={word}")
        predicted_tokens.append(word)
        
        trg_indices.append(pred_token)
        if pred_token == EOS_IDX: 
            break
            
    final_output = trg_tokenizer.decode(trg_indices)
    print(f"DEBUG: Final Output String: '{final_output}'")
    
    if not final_output.strip():
        return "[Model returned empty string. It might have predicted <eos> immediately.]"
        
    return final_output

# ==========================================
# 6. Routes
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    translation = ""
    original_text = ""
    
    if request.method == 'POST':
        original_text = request.form.get('source_text', '')
        try:
            translation = translate_sentence(original_text)
        except Exception as e:
            translation = f"Error: {e}"

    return render_template('index.html', translation=translation, original_text=original_text)

if __name__ == '__main__':
    app.run(debug=True, port=5000)