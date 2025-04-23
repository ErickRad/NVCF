from NN import Encoder, Decoder
from dataset import Loader

modeloEnc = Encoder()
modeloDec = Decoder()

dataset = Loader()

trainDataloader, testDataloader = dataset.load()