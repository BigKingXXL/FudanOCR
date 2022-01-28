import codecs
import torch
import torchvision
from .cdistnet.model.translator import Translator
from .cdistnet.model.model import CDistNet, build_CDistNet

WORD2IDX = None
IDX2WORD = None
CONFIG = None

def load_vocab(vocab=None, vocab_size=None):
    """
    Load vocab from disk. The fisrt four items in the vocab should be <PAD>, <UNK>, <S>, </S>
    """
    global WORD2IDX
    global IDX2WORD
    if WORD2IDX is not None and IDX2WORD is not None:
        return WORD2IDX, IDX2WORD
    assert vocab is not None and vocab_size is not None, "vocab is not precomputed"
    # print('Load set vocabularies as %s.' % vocab)
    vocab = [' ' if len(line.split()) == 0 else line.split()[0] for line in codecs.open(vocab, 'r', 'utf-8')]
    vocab = vocab[:vocab_size]
    assert len(vocab) == vocab_size
    WORD2IDX = {word: idx for idx, word in enumerate(vocab)}
    IDX2WORD = {idx: word for idx, word in enumerate(vocab)}
    
    return WORD2IDX, IDX2WORD

def get_parameter_number(net):
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return 'Trainable: {} M'.format(trainable_num/1000000)

def build_translator(cfg) -> Translator:
    model = build_CDistNet(cfg)
    en = get_parameter_number(model.transformer.encoder)
    de = get_parameter_number(model.transformer.decoder)
    print('encoder:{}\ndecoder:{}\n'.format(en,de))
    model_path = cfg.test.model_path
    print(f"laoding cdist model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    device = torch.device(cfg.test.device)
    model.to(device)
    model.eval()
    translator = Translator(cfg, model)
    load_vocab(cfg)
    return translator

def build_cdistnet(cfg) -> CDistNet:
    return build_CDistNet(cfg)


def make_prediction(model: Translator, input: torch.Tensor):
    image = input.clone()
    word2idx, idx2word = load_vocab()
    resize = torchvision.transforms.Resize((model.width, model.height), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
    image = resize(image)
    image = torch.unsqueeze(image, -1)
    image = image.permute(2, 3, 0, 1)
    image = image / 128. - 1.
    all_hyp, all_scores = model.translate_batch(image)

