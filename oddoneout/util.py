import torch


def cudaify(x):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            cuda = torch.device('cuda:2')
        else:
            cuda = torch.device('cuda:0')
        return x.cuda(cuda)
    else: 
        return x


def compare_tensors(t1, t2):
    t1 = cudaify(t1)
    t2 = cudaify(t2)
    return t1.shape == t2.shape and torch.allclose(t1, t2)


if torch.cuda.is_available():
    print("using gpu")
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    print("using cpu")
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
