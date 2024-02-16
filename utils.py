def accuracy(target, pred):
    acc = sum(target==pred)/len(target)
    return acc