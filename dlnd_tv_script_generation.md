
# TV Script Generation
In this project, you'll generate your own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  You'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network you'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
## Get the Data
The data is already provided for you.  You'll be using a subset of the original dataset.  It consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
```

## Explore the Data
Play around with `view_sentence_range` to view different parts of the data.


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 11492
    Number of scenes: 262
    Average number of sentences in each scene: 15.248091603053435
    Number of lines: 4257
    Average number of words in each line: 11.50434578341555
    
    The sentences 0 to 10:
    Moe_Szyslak: (INTO PHONE) Moe's Tavern. Where the elite meet to drink.
    Bart_Simpson: Eh, yeah, hello, is Mike there? Last name, Rotch.
    Moe_Szyslak: (INTO PHONE) Hold on, I'll check. (TO BARFLIES) Mike Rotch. Mike Rotch. Hey, has anybody seen Mike Rotch, lately?
    Moe_Szyslak: (INTO PHONE) Listen you little puke. One of these days I'm gonna catch you, and I'm gonna carve my name on your back with an ice pick.
    Moe_Szyslak: What's the matter Homer? You're not your normal effervescent self.
    Homer_Simpson: I got my problems, Moe. Give me another one.
    Moe_Szyslak: Homer, hey, you should not drink to forget your problems.
    Barney_Gumble: Yeah, you should only drink to enhance your social skills.
    
    


## Implement Preprocessing Functions
The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`


```python
import numpy as np
import problem_unittests as tests
from collections import Counter
def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: text for ii, text in enumerate(sorted_vocab)}
    vocab_to_int = {text: ii for ii, text in int_to_vocab.items()}
    
    return (vocab_to_int, int_to_vocab)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)
```

    Tests Passed


### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".

Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( . )
- Comma ( , )
- Quotation Mark ( " )
- Semicolon ( ; )
- Exclamation mark ( ! )
- Question mark ( ? )
- Left Parentheses ( ( )
- Right Parentheses ( ) )
- Dash ( -- )
- Return ( \n )

This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word. Make sure you don't use a token that could be confused as a word. Instead of using the token "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    tokenize = {}
    tokenize['.']="||period||"
    tokenize[',']="||comma||"
    tokenize['"']="||quotation_mark||"
    tokenize[';']="||semicolon||"
    tokenize['!']="||exclamation_mark||"
    tokenize['?']="||question_mark||"
    tokenize['(']="||left_parantheses||"
    tokenize[')']="||right_parantheses||"
    tokenize['--'] = "||dash||"
    tokenize['\n'] = "||return||"
    
    return tokenize

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)
```

    Tests Passed


## Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

## Build the Neural Network
You'll build the components necessary to build a RNN by implementing the following functions below:
- get_inputs
- get_init_cell
- get_embed
- build_rnn
- build_nn
- get_batches

### Check the Version of TensorFlow and Access to GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.0
    Default GPU Device: /gpu:0



```python
tf.test.gpu_device_name()
```




    '/gpu:0'



### Input
Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.  It should create the following placeholders:
- Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
- Targets placeholder
- Learning Rate placeholder

Return the placeholders in the following tuple `(Input, Targets, LearningRate)`


```python
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    input = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return (input,targets,learning_rate)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)
```

    Tests Passed


### Build RNN Cell and Initialize
Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
- The Rnn size should be set using `rnn_size`
- Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
    - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the cell and initial state in the following tuple `(Cell, InitialState)`


```python
# 这里的模型需要不断的调整
def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    #batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    lstm1 = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm2 = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm3 = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm1,lstm2,lstm3])
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')
    return cell, initial_state
    
#     lstm1 = tf.contrib.rnn.BasicLSTMCell(rnn_size)
#     lstm2 = tf.contrib.rnn.BasicLSTMCell(rnn_size)
#     cell = tf.contrib.rnn.MultiRNNCell([lstm1,lstm2])
#     initial_state = tf.identity(cell.zero_state(batch_size, tf.float32), "initial_state")
#     return (cell, initial_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)
```

    Tests Passed


### Word Embedding
Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.


```python
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)
```

    Tests Passed


### Build RNN
You created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
- Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
 - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the outputs and final_state state in the following tuple `(Outputs, FinalState)` 


```python
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype='float32')
    final_state = tf.identity(final_state, name='final_state')
    return outputs, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)
```

    Tests Passed


### Build the Neural Network
Apply the functions you implemented above to:
- Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.
- Build RNN using `cell` and your `build_rnn(cell, inputs)` function.
- Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.

Return the logits and final state in the following tuple (Logits, FinalState) 


```python
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    embed = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, embed)
    weights = tf.truncated_normal_initializer(stddev=0.1)
    logits = tf.contrib.layers.fully_connected(outputs, 
                                               vocab_size, 
                                               activation_fn=None,
                                               weights_initializer=weights,
                                               biases_initializer=tf.zeros_initializer())
    return logits, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)
```

    Tests Passed


### Batches
Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
- The first element is a single batch of **input** with the shape `[batch size, sequence length]`
- The second element is a single batch of **targets** with the shape `[batch size, sequence length]`

If you can't fill the last batch with enough data, drop the last batch.

For exmple, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 2, 3)` would return a Numpy array of the following:
```
[
  # First Batch
  [
    # Batch of Input
    [[ 1  2  3], [ 7  8  9]],
    # Batch of targets
    [[ 2  3  4], [ 8  9 10]]
  ],
 
  # Second Batch
  [
    # Batch of Input
    [[ 4  5  6], [10 11 12]],
    # Batch of targets
    [[ 5  6  7], [11 12 13]]
  ]
]
```


```python
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    n_batches = int(len(int_text) / (batch_size * seq_length))
    # Drop the last few characters to make only full batches
    xdata = np.array(int_text[: n_batches * batch_size * seq_length])
    ydata = np.array(int_text[1: n_batches * batch_size * seq_length + 1])
    ydata[-1] = xdata[0]

    x_batches = np.split(xdata.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), n_batches, 1)
    return np.array(list(zip(x_batches, y_batches)))

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)
```

    Tests Passed


## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `num_epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `embed_dim` to the size of the embedding.
- Set `seq_length` to the length of sequence.
- Set `learning_rate` to the learning rate.
- Set `show_every_n_batches` to the number of batches the neural network should print progress.


```python
# Number of Epochs
num_epochs = 1000
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256
# Embedding Dimension Size
embed_dim = 600
# Sequence Length
seq_length = 50
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 64

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
```

## Train
Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forms](https://discussions.udacity.com/) to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/10   train_loss = 8.827
    Epoch   6 Batch    4/10   train_loss = 5.809
    Epoch  12 Batch    8/10   train_loss = 5.511
    Epoch  19 Batch    2/10   train_loss = 5.055
    Epoch  25 Batch    6/10   train_loss = 4.761
    Epoch  32 Batch    0/10   train_loss = 4.520
    Epoch  38 Batch    4/10   train_loss = 4.265
    Epoch  44 Batch    8/10   train_loss = 4.099
    Epoch  51 Batch    2/10   train_loss = 4.016
    Epoch  57 Batch    6/10   train_loss = 3.947
    Epoch  64 Batch    0/10   train_loss = 3.810
    Epoch  70 Batch    4/10   train_loss = 3.662
    Epoch  76 Batch    8/10   train_loss = 3.564
    Epoch  83 Batch    2/10   train_loss = 3.414
    Epoch  89 Batch    6/10   train_loss = 3.538
    Epoch  96 Batch    0/10   train_loss = 3.219
    Epoch 102 Batch    4/10   train_loss = 3.273
    Epoch 108 Batch    8/10   train_loss = 2.996
    Epoch 115 Batch    2/10   train_loss = 2.954
    Epoch 121 Batch    6/10   train_loss = 2.874
    Epoch 128 Batch    0/10   train_loss = 2.678
    Epoch 134 Batch    4/10   train_loss = 2.840
    Epoch 140 Batch    8/10   train_loss = 2.539
    Epoch 147 Batch    2/10   train_loss = 2.430
    Epoch 153 Batch    6/10   train_loss = 2.372
    Epoch 160 Batch    0/10   train_loss = 2.271
    Epoch 166 Batch    4/10   train_loss = 2.186
    Epoch 172 Batch    8/10   train_loss = 2.071
    Epoch 179 Batch    2/10   train_loss = 1.997
    Epoch 185 Batch    6/10   train_loss = 1.885
    Epoch 192 Batch    0/10   train_loss = 1.849
    Epoch 198 Batch    4/10   train_loss = 1.746
    Epoch 204 Batch    8/10   train_loss = 1.579
    Epoch 211 Batch    2/10   train_loss = 1.577
    Epoch 217 Batch    6/10   train_loss = 1.579
    Epoch 224 Batch    0/10   train_loss = 1.426
    Epoch 230 Batch    4/10   train_loss = 1.424
    Epoch 236 Batch    8/10   train_loss = 1.271
    Epoch 243 Batch    2/10   train_loss = 1.347
    Epoch 249 Batch    6/10   train_loss = 1.111
    Epoch 256 Batch    0/10   train_loss = 1.087
    Epoch 262 Batch    4/10   train_loss = 1.103
    Epoch 268 Batch    8/10   train_loss = 0.954
    Epoch 275 Batch    2/10   train_loss = 1.058
    Epoch 281 Batch    6/10   train_loss = 0.906
    Epoch 288 Batch    0/10   train_loss = 0.807
    Epoch 294 Batch    4/10   train_loss = 0.956
    Epoch 300 Batch    8/10   train_loss = 0.755
    Epoch 307 Batch    2/10   train_loss = 0.667
    Epoch 313 Batch    6/10   train_loss = 0.705
    Epoch 320 Batch    0/10   train_loss = 0.658
    Epoch 326 Batch    4/10   train_loss = 0.589
    Epoch 332 Batch    8/10   train_loss = 0.573
    Epoch 339 Batch    2/10   train_loss = 0.526
    Epoch 345 Batch    6/10   train_loss = 0.563
    Epoch 352 Batch    0/10   train_loss = 0.541
    Epoch 358 Batch    4/10   train_loss = 0.470
    Epoch 364 Batch    8/10   train_loss = 0.382
    Epoch 371 Batch    2/10   train_loss = 0.381
    Epoch 377 Batch    6/10   train_loss = 0.439
    Epoch 384 Batch    0/10   train_loss = 0.449
    Epoch 390 Batch    4/10   train_loss = 0.320
    Epoch 396 Batch    8/10   train_loss = 0.262
    Epoch 403 Batch    2/10   train_loss = 0.250
    Epoch 409 Batch    6/10   train_loss = 0.244
    Epoch 416 Batch    0/10   train_loss = 0.241
    Epoch 422 Batch    4/10   train_loss = 0.556
    Epoch 428 Batch    8/10   train_loss = 0.323
    Epoch 435 Batch    2/10   train_loss = 0.200
    Epoch 441 Batch    6/10   train_loss = 0.180
    Epoch 448 Batch    0/10   train_loss = 0.161
    Epoch 454 Batch    4/10   train_loss = 0.160
    Epoch 460 Batch    8/10   train_loss = 0.151
    Epoch 467 Batch    2/10   train_loss = 0.150
    Epoch 473 Batch    6/10   train_loss = 0.316
    Epoch 480 Batch    0/10   train_loss = 0.379
    Epoch 486 Batch    4/10   train_loss = 0.147
    Epoch 492 Batch    8/10   train_loss = 0.126
    Epoch 499 Batch    2/10   train_loss = 0.120
    Epoch 505 Batch    6/10   train_loss = 0.112
    Epoch 512 Batch    0/10   train_loss = 0.104
    Epoch 518 Batch    4/10   train_loss = 0.103
    Epoch 524 Batch    8/10   train_loss = 0.097
    Epoch 531 Batch    2/10   train_loss = 0.095
    Epoch 537 Batch    6/10   train_loss = 0.090
    Epoch 544 Batch    0/10   train_loss = 0.085
    Epoch 550 Batch    4/10   train_loss = 0.086
    Epoch 556 Batch    8/10   train_loss = 0.082
    Epoch 563 Batch    2/10   train_loss = 0.081
    Epoch 569 Batch    6/10   train_loss = 0.076
    Epoch 576 Batch    0/10   train_loss = 0.073
    Epoch 582 Batch    4/10   train_loss = 0.075
    Epoch 588 Batch    8/10   train_loss = 0.072
    Epoch 595 Batch    2/10   train_loss = 0.071
    Epoch 601 Batch    6/10   train_loss = 0.067
    Epoch 608 Batch    0/10   train_loss = 0.065
    Epoch 614 Batch    4/10   train_loss = 0.067
    Epoch 620 Batch    8/10   train_loss = 0.064
    Epoch 627 Batch    2/10   train_loss = 0.064
    Epoch 633 Batch    6/10   train_loss = 0.061
    Epoch 640 Batch    0/10   train_loss = 0.059
    Epoch 646 Batch    4/10   train_loss = 0.061
    Epoch 652 Batch    8/10   train_loss = 0.059
    Epoch 659 Batch    2/10   train_loss = 0.058
    Epoch 665 Batch    6/10   train_loss = 0.057
    Epoch 672 Batch    0/10   train_loss = 0.055
    Epoch 678 Batch    4/10   train_loss = 0.057
    Epoch 684 Batch    8/10   train_loss = 0.056
    Epoch 691 Batch    2/10   train_loss = 0.063
    Epoch 697 Batch    6/10   train_loss = 0.523
    Epoch 704 Batch    0/10   train_loss = 0.249
    Epoch 710 Batch    4/10   train_loss = 0.090
    Epoch 716 Batch    8/10   train_loss = 0.062
    Epoch 723 Batch    2/10   train_loss = 0.058
    Epoch 729 Batch    6/10   train_loss = 0.056
    Epoch 736 Batch    0/10   train_loss = 0.054
    Epoch 742 Batch    4/10   train_loss = 0.055
    Epoch 748 Batch    8/10   train_loss = 0.054
    Epoch 755 Batch    2/10   train_loss = 0.052
    Epoch 761 Batch    6/10   train_loss = 0.051
    Epoch 768 Batch    0/10   train_loss = 0.050
    Epoch 774 Batch    4/10   train_loss = 0.051
    Epoch 780 Batch    8/10   train_loss = 0.051
    Epoch 787 Batch    2/10   train_loss = 0.049
    Epoch 793 Batch    6/10   train_loss = 0.049
    Epoch 800 Batch    0/10   train_loss = 0.048
    Epoch 806 Batch    4/10   train_loss = 0.049
    Epoch 812 Batch    8/10   train_loss = 0.049
    Epoch 819 Batch    2/10   train_loss = 0.047
    Epoch 825 Batch    6/10   train_loss = 0.047
    Epoch 832 Batch    0/10   train_loss = 0.046
    Epoch 838 Batch    4/10   train_loss = 0.048
    Epoch 844 Batch    8/10   train_loss = 0.048
    Epoch 851 Batch    2/10   train_loss = 0.046
    Epoch 857 Batch    6/10   train_loss = 0.046
    Epoch 864 Batch    0/10   train_loss = 0.045
    Epoch 870 Batch    4/10   train_loss = 0.047
    Epoch 876 Batch    8/10   train_loss = 0.047
    Epoch 883 Batch    2/10   train_loss = 0.045
    Epoch 889 Batch    6/10   train_loss = 0.045
    Epoch 896 Batch    0/10   train_loss = 0.044
    Epoch 902 Batch    4/10   train_loss = 0.046
    Epoch 908 Batch    8/10   train_loss = 0.046
    Epoch 915 Batch    2/10   train_loss = 0.044
    Epoch 921 Batch    6/10   train_loss = 0.045
    Epoch 928 Batch    0/10   train_loss = 0.043
    Epoch 934 Batch    4/10   train_loss = 0.045
    Epoch 940 Batch    8/10   train_loss = 0.045
    Epoch 947 Batch    2/10   train_loss = 0.044
    Epoch 953 Batch    6/10   train_loss = 0.044
    Epoch 960 Batch    0/10   train_loss = 0.043
    Epoch 966 Batch    4/10   train_loss = 0.045
    Epoch 972 Batch    8/10   train_loss = 0.045
    Epoch 979 Batch    2/10   train_loss = 0.043
    Epoch 985 Batch    6/10   train_loss = 0.044
    Epoch 992 Batch    0/10   train_loss = 0.042
    Epoch 998 Batch    4/10   train_loss = 0.044
    Model Trained and Saved


稍稍过拟合.

## Save Parameters
Save `seq_length` and `save_dir` for generating a new TV script.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()
```

## Implement Generate Functions
### Get Tensors
Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
- "input:0"
- "initial_state:0"
- "final_state:0"
- "probs:0"

Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)` 


```python
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    input_tensor = loaded_graph.get_tensor_by_name('input:0')
    initial_state_tensor = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state_tensor = loaded_graph.get_tensor_by_name('final_state:0')
    probs_tensor = loaded_graph.get_tensor_by_name('probs:0')
    
    return input_tensor, initial_state_tensor, final_state_tensor, probs_tensor


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)
```

    Tests Passed


### Choose Word
Implement the `pick_word()` function to select the next word using `probabilities`.


```python
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    return int_to_vocab[np.argmax(probabilities)]


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)
```

    Tests Passed


## Generate TV Script
This will generate the TV script for you.  Set `gen_length` to the length of TV script you want to generate.


```python
gen_length = 300
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)
```

    moe_szyslak: oh, so you're looking for a mr. smithers, eh? first name, waylon, is it?(suddenly vicious) listen to me, you... when i catch you, i'm going to pull out your eyes and shove 'em up your alley!
    homer_simpson:(terrified noise) can't believe there's me we'd all everyone, fellas i even ya, i'm just a rough cheer, there's what you gave you the greatest room the world, that's the wire.(singing) too happened, even you your last sickens, i'm one.
    
    
    grampa_simpson: here's the beauty part with me next military saturday food better.
    barney_gumble: hey, i'm sorry, it'll get a guy(indicates voice) who's a guess... when i had something with my fans? but homer, all right i should steal your new air conditioner.
    moe_szyslak: shut a keys today.
    bart_simpson:(offended) are that dumb?
    chief_wiggum:(singing) dad, you were better this bar about if they been away more there, and you can good friends. maybe i can believe our guy barney been a duff?
    moe_szyslak: this is a great product, this is smooth, glass whale, who's time, and cecil hampstead-on-cecil-cecil.
    moe_szyslak: so, uh, you little rutabaga brain, i'll take your eyeball and me that that a dank. the dank!
    homer_simpson:(quietly) i'm so take anything to have to take a money.(laughs)
    homer_simpson:(excited) yeah. you've got me-- she since duff hounds to


# The TV Script is Nonsensical
It's ok if the TV script doesn't make any sense.  We trained on less than a megabyte of text.  In order to get good results, you'll have to use a smaller vocabulary or get more data.  Luckly there's more data!  As we mentioned in the begging of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  We didn't have you train on all the data, because that would take too long.  However, you are free to train your neural network on all the data.  After you complete the project, of course.
# Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_tv_script_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
