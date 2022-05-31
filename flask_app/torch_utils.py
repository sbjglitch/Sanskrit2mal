import torch 
import torch.nn as  nn
import torchvision.transforms as transform
import torch.nn.functional as F
import inltk 
from inltk.inltk import setup
from inltk.inltk import tokenize
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_text = 'धृतराष्ट्र उवाच धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय।।'

mal_lines = []
sanskrit_lines = []
with open ("C:\\Users\\Glitch\Sanskrit2mal\\parallel-corpus\\bhagavadgita_malayalam.txt",'r', encoding="utf8") as f:
  mal_lines.extend([x.replace('\n', '') for x in f.readlines()])

with open ('C:\\Users\\Glitch\\Sanskrit2mal\\parallel-corpus\\bhagvadgita_sanskrit.txt','r',encoding="utf8") as f:
  sanskrit_lines.extend([x.replace('\n', '') for x in f.readlines()])

train_text_sa = sanskrit_lines[:701]
train_text_mal = mal_lines[:701]

dev_text_sa = sanskrit_lines[:701]
dev_text_mal = mal_lines[:701]


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    # Forming pairs of sentences
    pairs = [[train_text_sa[i], train_text_mal[i]] for i in range(len(train_text_sa))]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('sanskrit', 'mal', False)

def indexesFromSentence(lang, sentence):
    res = []
    for word in sentence.split(' '):
      if word not in lang.word2index.keys():
        res.append(list(lang.word2index.values()))    
      else:
        res.append(lang.word2index[word])
    return res

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

hidden_size = 256

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# %cd C:\Users\Glitch\Sanskrit2mal


MAX_LENGTH = 200

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


PATH = 'C:\\Users\\Glitch\\Sanskrit2mal\\trained_decoder.pth'        

decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)
decoder.load_state_dict(torch.load(PATH))
decoder.eval()
decoder.to(device)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def translator(text):
    #test_text = 'धृतराष्ट्र उवाच धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय।।'
    PATH = 'C:\\Users\\Glitch\\Sanskrit2mal\\trained_encoder.pth'        
    encoder = EncoderRNN(input_lang.n_words, hidden_size)
    encoder.load_state_dict(torch.load(PATH))
    encoder.eval()
    encoder.to(device)

    PATH = 'C:\\Users\\Glitch\\Sanskrit2mal\\trained_decoder.pth'        
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)
    decoder.load_state_dict(torch.load(PATH))
    decoder.eval()
    decoder.to(device)
    
    output_words, attentions = evaluate(encoder, decoder, text)
    output_sentence = ' '.join(output_words)
    with open('C:\\Users\\Glitch\\Sanskrit2mal\\out.txt', 'w', encoding="utf8") as f:
        f.write(output_sentence + '\n')
    return output_sentence
    #print(output_sentence)

#translator (model1,model2)   