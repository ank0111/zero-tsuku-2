from time_layers import *
from base_model import BaseModel
import numpy as np


class BetterRnnlm(BaseModel):
    '''
    LSTMレイヤを2層利用し、各層にDropoutを使うモデル
    '''

    def __init__(self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5):
        V, D, O = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')
        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTMWithDropout(lstm_Wx1, lstm_Wh1, lstm_b1, True, dropout_ratio),
            TimeDropout(dropout_ratio),
            TimeLSTMWithDropout(lstm_Wx2, lstm_Wh2, lstm_b2, True, dropout_ratio),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)
        ]
