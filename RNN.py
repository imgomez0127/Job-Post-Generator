import os
import json
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Embedding
from tensorflow.keras.models import save_model
import numpy as np
from jobparser import importAllJsons
tf.enable_eager_execution()
def generate_text(model,start_str,charToIdx,idxToChar):
    charCount = 200
    input_eval = [charToIdx[c] for c in start_str]
    input_eval = tf.expand_dims(input_eval,0)
    text_generated = []
    temperature = 1.0
    model.reset_states()
    for i in range(charCount):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions,0)
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id],0)
        text_generated.append(idxToChar[predicted_id])
    return (start_str + ''.join(text_generated))
def build_model(charSetSize,embeddingDim,rnnUnits,batchSize):
    if tf.test.is_gpu_available():
        rnn = keras.layers.CuDNNGRU
    else:
        import functools 
        rnn = functools.partial(keras.layers.GRU,recurrent_activation="sigmoid")
    model= keras.Sequential()
    model.add(Embedding(charSetSize,embeddingDim,batch_input_shape=[batchSize,None]))
    model.add(rnn(rnnUnits,return_sequences=True,recurrent_initializer="glorot_uniform",stateful=True))
    model.add(Dense(charSetSize))
    return model
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text,target_text
jsonsLst = importAllJsons()
charSet = set()
text = ""
for jsonItem in jsonsLst:
    text += jsonItem["body"].strip().lower() +"\n\n"
print(text)
for JSON in jsonsLst:
    charSet = charSet.union(set(JSON["body"]))
charToIdx = {c:i for i,c in enumerate(charSet)}
idxToChar = {i:c for i,c in enumerate(charSet)}
text_as_int = np.array([charToIdx[c] for c in text])
seq_length = 100
examples_per_epoch = len(text_as_int)//seq_length
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
for i in char_dataset.take(5):
    print(idxToChar[i.numpy()])
sequences = char_dataset.batch(seq_length+1,drop_remainder=True)
dataset = sequences.map(split_input_target)
for input_example, target_example in  dataset.take(1):
    print ('Input data: ', repr(''.join(map(lambda c: idxToChar[c],input_example.numpy()))))
    print ('Target data: ', repr(''.join(map(lambda c: idxToChar[c],target_example.numpy()))))

print(charToIdx)
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print(input_idx.numpy())
    print("  input: {} ({:s})".format(input_idx, repr(idxToChar[input_idx.numpy()])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idxToChar[target_idx.numpy()])))
BATCH_SIZE = 5
steps_per_epoch = examples_per_epoch//BATCH_SIZE
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)
charSize = len(charSet)
embedding_dim = 256
rnn_units = 1024
print("DAWNIUDNA")
model = build_model(charSize,embedding_dim,rnn_units,BATCH_SIZE)
print("DANWOIDNAWDNIAWONDIAWNO")
for input_example_batch, target_example_batch in dataset.take(1): 
    print("hello")
    example_batch_predictions = model(input_example_batch)
    print("yes")
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
model.summary()
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
print(sampled_indices)
print("Untrained prediction: ",end="")
print(repr("".join(list(map(lambda x: idxToChar[x], sampled_indices)))))
def categoricalLoss(labels,logits):
    return keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True)
print(categoricalLoss(target_example_batch,example_batch_predictions))
model.compile(optimizer="Adam",loss=categoricalLoss)
checkpoint_dir= "./test"
checkpoint_prefix = os.path.join(checkpoint_dir,"ckpt_{epoch}")
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only = True)
EPOCHS = 60
history = model.fit(dataset.repeat(), epochs = EPOCHS,steps_per_epoch=steps_per_epoch,callbacks=[checkpoint_callback])
model = build_model(charSize, embedding_dim, rnn_units, batchSize=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
print(generate_text(model,"icims",charToIdx,idxToChar))
save_model(model,"models/RNNModel.h5")
modelParamsPath = "models/modelparams.json"
modelParams = {"charToIdx":charToIdx,"idxToChar":idxToChar}
with open(modelParamsPath,"w") as f:
    json.dump(modelParams,f)
