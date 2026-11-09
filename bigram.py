words = open("names.txt", "r").read().splitlines()

print(words[:1])
b = {}
for w in words:
    chrs = ["."] + list(w) + ["."] #start and end characters to the word
    for cr1, cr2 in zip(chrs, chrs[1:]): # pairs adjacent letters with one another
        bigram = (cr1, cr2)
        b[bigram] = b.get(bigram, 0) + 1 # counts the number of bigrams so far, stores in b. if none, default 0.

# covert integers / chrs into something we can feed into a neural net by turning it into a vector with a certain dimension = 1.


import torch
N = torch.zeros((27, 27), dtype=torch.int32) # create 27 x 27 matrix of zeros

chars = sorted(list(set(''.join(words)))) # find all unique characters in our dataset (set has no duplicates)
stoi = {s:i+1 for i,s in enumerate(chars)} # make dictionary indexing each character with a letter. string to integer.
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()} # integer to string


for (ch1, ch2), count in b.items(): # bigrams again
    ix1 = stoi[ch1] # convert bigram to numbers
    ix2 = stoi[ch2]
    N[ix1, ix2] = count # add one to the numerical position of bigram in matrix. row follows column


p = N[0].float() # N(0) is the first row of frequencies. I.E: the frequencies that a, b, c, d follows a in our dataset.
p = p / p.sum() # divide by total frequencies for probability

P = (N+1).float() # set p to our matrix of frequencies and add 1 for smoothing
P /= P.sum(1, keepdims=True) # built into pytorch: divide every element in every row by each row's total. /= applies this to every row.

g = torch.Generator().manual_seed(2147483647)

#for i in range(10): # GENERATING SOME NAMES
#  out = []
#  ix = 0
#  while True:
#    p = P[ix]
#    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() # RANDOM SAMPLING OF INDEXES ACCORDING TO PROBS
#    out.append(itos[ix])
#    if ix == 0:
#      break
#  print(''.join(out))

# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# log(a*b*c) = log(a) + log(b) + log(c)

log_likelihood = 0.0
n = 0

#for w in ["andrejq"]: #loss for a given name
for w in words: # loss for the entire dataset
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob) # take the log of each probability the model assigns to a bigram in the word
    log_likelihood += logprob # add all these logprobs together to get the total logprob of the word. We want this to be as high as possible. (minimise -logprob)
    n += 1 # in reality we take the average of the -logprob. This is to stop longer words having a higher loss just by virtue of being longer (more chances to get things wrong)
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')




# create the training set of bigrams (x,y)
xs, ys = [], [] # x = input. y = target to be predicted

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]): # just the first word "emma"
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1) # we have two lists of input and target
    ys.append(ix2)

xs = torch.tensor(xs) # convert these to tensors.
ys = torch.tensor(ys)

# then we need to convert these integers into a form that a neural network can deal with (which only uses 0-1). We use a 27-vector! onehot.

import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes = 27).float() #puts each unique integer into an index within a 5 * 27 matrix (row, column). our word as a matrix.

w = torch.randn((27, 1), requires_grad=True) # produces a row vector of numbers drawn from the normal distribution. Our starting weights.

xenc @ w # @ is matmul operator. row x column = single number. column x row = matrix. matrix * column = column. guess we want col x row to get more info out.
# this gives us a 5 col vector

w = torch.randn((27, 27), requires_grad=True) # 27 neurons with 27 weights

print(xenc @ w) # 5 * 27 matrix. (r, c) tells us the activation of the cth neuron on the rth character
# but we need the outputs to be something like probabilities. positive and sum to 1. we pretend that the neural
# net outputs are log counts. So, we exponentiate them to get the real counts, and then normalize to get a probability. LOGITS = LOGCOUNTS.

# FORWARD PASS
for i in range(0,200):
    logits = (xenc @ w) # exponentiate to get LOGITS
    loss = logits.sum()


    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)

    probs /= probs.sum(1, keepdims=True) # normalise for each row. so each entry us the prob each neuron assigns to each character
    # THIS WHOLE PROCESS IS CALLED SOFTMAX - getting probability distributions out of neural networks.
    # Now, we just calcuate the loss with log liklihoods (what prob it assigns to each bigram in the data) and backprop the weights based on that loss.

    loss = -probs[torch.arange(len(xs)), ys].log().mean() #our predicted probabilities for the character which actually followed the xs. np.arrage just has the characters in order.
    # we take the log and mean and change sign for the neg avg log prob.


    # BACKWARD PASS
    w.grad = None
    loss.backward()
    w.data += -50* w.grad
    print(loss)
