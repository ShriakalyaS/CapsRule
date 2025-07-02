from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math
from Rule_Extraction_and_or import extract_rules_boundary, mean, std
from Rule_Evaluation import evaluate_rules_boundary
from Rule_Validation import validate_rules

from Squash_s import squash, safe_norm
from sklearn.preprocessing import Normalizer, minmax_scale
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
from Performance_measures import confusion_metrics_tf, Micro_calculate_measures, Macro_calculate_measures_basic, Macro_calculate_measures_tf

'''
Global Scope
'''
batch_indx = 0
init_sigma = 1.0
'''
CONSTANTS
'''
# # @@@@@@@@@@@@@@@@@@FFCN Constants@@@@@@@@@@@@@@@@@@@@@@@@
# &&&&&&&&&&&&&&&&CICDOS2019-12 Features&&&&&&&&&&&&&&&&&&&
batch_size = 64
N_CLASSSES = 4
num_layers = 5
size_layers = [25, 20, 15, 8, N_CLASSSES] #output_dimesion for each layer
caps_dim = [30, 25, 20, 15, 10]  # deeper capsules per layer

caps_dim_input = 1
input = 30
Routing_Iter = 3
LRATE = 0.001
n_epochs = 50
loss_thresh = 0.1
restore_checkpoint = True
# $$$$$$$$$$$$$$$$$Decoder Constants$$$$$$$$$$$$$$$$$$$$$$$
n_hidden1 = 5
n_hidden2 = 15
n_hidden3 = 20
n_output = input
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def condition(d, caps_predicted, routing_weights, raw_weights, caps1_n,caps2_n, counter):
    return tf.less(counter, Routing_Iter)

def loop_body(caps_output, caps_predicted, routing_weights, raw_weights, caps1_n, caps2_n, counter):
    '''to do elementwise matrix multiplication'''
    '''uj|i * caps2_predicted shape=(caps1_n_caps, caps2_n_caps, caps_dim, caps_dim)'''
    weighted_predictions = tf.multiply(routing_weights, caps_predicted, name="weighted_predictions")
    '''s'''
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")
    # '''Squash the output 𝐯𝑗=squash(𝐬𝑗) '''
    caps_output = squash(weighted_sum, axis=-2, name="caps_output")
    # caps_output = tf.nn.relu(BN, name="caps_output")
    '''let's measure how close each predicted vector  𝐮̂ 𝑗|𝑖  is to the actual output vector
      𝐯𝑗  by computing their scalar product  𝐮̂ 𝑗|𝑖⋅𝐯𝑗'''
    '''We need to make the dimensions match'''
    caps_output_tiled = tf.tile(caps_output, [1, caps1_n, 1, 1, 1], name="caps_output_tiled")
    agreement = tf.matmul(caps_predicted, caps_output_tiled, transpose_a=True, name="agreement")

    '''We can now update the raw routing weights  𝑏𝑖,𝑗  by simply adding the scalar product  
    𝐮̂ 𝑗|𝑖⋅𝐯𝑗  we just computed:  𝑏𝑖,𝑗←𝑏𝑖,𝑗+𝐮̂ 𝑗|𝑖⋅𝐯𝑗'''
    raw_weights = tf.add(raw_weights, agreement, name="raw_weights")

    '''repeated exactly as round 1'''
    '''Ci: coupling coefficients'''
    routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

    return caps_output, caps_predicted, routing_weights, raw_weights, caps1_n, caps2_n, tf.add(counter, 1)
def caps_layer(input_size, output_size, caps_dim_input,caps_dim_output):

    '''
    Computation Graph
    '''

    hidden_layer = {
                      'weights': tf.Variable(tf.random_normal(shape=(1, input_size, output_size, caps_dim_output, caps_dim_input)
                                                              , stddev=init_sigma, mean=0.0, dtype=tf.float32, name='weights')
                                             , name='weights'),
                      'raw_weights': tf.Variable(tf.zeros([batch_size, input_size, output_size, 1, 1],
                                                          dtype=np.float32, name="raw_weights")),
                      }
    return hidden_layer

def neural_network_model(data):

    input_size=input
    caps_dim_input=1

    '''Expand the input capsule (u) three times to have the shape of u=(input, caps_layer_1, input_dim=1, 1)'''
    input_expanded_1 = tf.expand_dims(data, -1, name="input_expanded_1")
    input_expanded_2 = tf.expand_dims(input_expanded_1, 2, name="input_expanded_2")
    input_expanded = tf.expand_dims(input_expanded_2, 3, name="input_expanded")

    counter = tf.constant(1)
    h = list()
    a_lst = list()
    b_lst = list()
    res_lst = list()

    for i in range(num_layers):
        h.append(caps_layer(input_size, size_layers[i], caps_dim_input,caps_dim[i]))
        with tf.name_scope("routing_by_agreement"):
            W_tiled = tf.tile(h[i]['weights'], [batch_size, 1, 1, 1, 1], name="W_tiled")
            input_expanded_tiled = tf.tile(input_expanded, [1, 1, size_layers[i], 1, 1], name="input_expanded_tiled")
            caps_predicted = tf.matmul(W_tiled, input_expanded_tiled, name="caps_predicted")
            ''' Add routing weights ci = softmax(bi)'''
            routing_weights = tf.nn.softmax(h[i]['raw_weights'], dim=2, name="routing_weights")
            result = tf.Variable(tf.zeros([batch_size, 1, size_layers[i], caps_dim[i], 1], dtype=np.float32, name="result"))
            result, a, b, c, d, e, f = tf.while_loop(condition, loop_body, [result, caps_predicted, routing_weights,
                                                                          h[i]['raw_weights'], input_size,size_layers[i],counter])
            pred_vector = tf.squeeze(a, axis=[4])
            a_lst.append(pred_vector)  # Caps-predicted list
            cpl_coeff = tf.squeeze(b, axis=[3, 4], name="cpl_coeff")
            b_lst.append(cpl_coeff)  # Routing weights list
            caps_out = tf.squeeze(result, axis=[1, -1])
            res_lst.append(caps_out)  # Caps-out list
            result_sq = tf.squeeze(result, axis=[1], name="result_sq")
            input_expanded = tf.expand_dims(result_sq, 2, name="input_expanded")
        input_size = size_layers[i]
        caps_dim_input = caps_dim[i]

    y_proba = safe_norm(result, axis=-2, name="y_proba")
    y_pred = tf.squeeze(y_proba, axis=[1, 3], name="y_pred")

    return data, y_pred, b_lst,a_lst, res_lst, result, W_tiled

def reconstruction(y, y_pred, caps_out, mask_with_labels):
    '''We need a placeholder to tell TensorFlow whether we want to mask the output vectors based on the labels (True)
    or on the predictions (False, the default):'''

    reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                     lambda: y,  # if True
                                     lambda: y_pred,  # if False
                                     name="reconstruction_targets")
    ''' Reconstruction_target is equal to 1.0 for the target class, and 0.0 for the other classes,'''

    reconstruction_mask_reshaped = tf.reshape(
        reconstruction_targets, [-1, 1, N_CLASSSES, 1, 1],
        name="reconstruction_mask_reshaped")

    caps_output_masked = tf.multiply(
        caps_out, reconstruction_mask_reshaped,
        name="caps_output_masked")

    decoder_input = tf.reshape(caps_output_masked,
                               [-1, size_layers[-1] * caps_dim[-1]],
                               name="decoder_input")

    return decoder_input

def decoder(decoder_input):
    with tf.name_scope("decoder"):
        hidden1 = tf.keras.layers.Dense(n_hidden1, activation=tf.nn.relu, name="hidden1")(decoder_input)
        hidden2 = tf.keras.layers.Dense(n_hidden2, activation=tf.nn.relu, name="hidden2")(hidden1)
        hidden3 = tf.keras.layers.Dense(n_hidden3, activation=tf.nn.relu, name="hidden3")(hidden2)
        decoder_output =tf.keras.layers.Dense(n_output,activation=tf.nn.sigmoid,name="decoder_output")(hidden3)
        
        ''''
        hidden1 = tf.keras.layers.Dense(decoder_input, n_hidden1,
                                  activation=tf.nn.relu,
                                  name="hidden1")
        hidden2 = tf.keras.layers.Dense(hidden1, n_hidden2,
                                  activation=tf.nn.relu,
                                  name="hidden2")
        hidden3 = tf.keras.layers.Dense(hidden2, n_hidden3,
                                  activation=tf.nn.relu,
                                  name="hidden3")
        decoder_output =tf.keras.layers.Dense(hidden3, n_output,
                                         activation=tf.nn.sigmoid,
                                         name="decoder_output")
        '''

    return decoder_output

import numpy as np


import numpy as np

def next_batch(data, batch_indx, batch_size):
    col_num = data.shape[1]  # Get actual number of columns dynamically

    if data.empty:
        raise ValueError("DataFrame is empty. Check dataset loading.")

    if col_num < 2:
        raise ValueError(f"Not enough columns in data. Found {col_num}, expected at least 2.")

    try:
        # Extract feature columns (all except the last)
        list_data = np.array(data.iloc[:, :col_num - 1])

        # Extract labels (last column)
        list_labels = np.array(data.iloc[:, col_num - 1], dtype=int)

        # Ensure labels are valid for one-hot encoding
        if np.max(list_labels) >= N_CLASSSES or np.min(list_labels) < 0:
            raise ValueError(f"Label values out of range (0 to {N_CLASSSES-1}). Found labels: {np.unique(list_labels)}")

        # Convert to one-hot encoding
        onehot_labels = np.zeros((len(list_labels), N_CLASSSES))
        onehot_labels[np.arange(len(list_labels)), list_labels] = 1
        list_labels = onehot_labels

        # Reset batch index if it exceeds dataset length
        if batch_indx + batch_size > len(data):
            batch_indx = 0  

        # Get batch data
        lst_data = list_data[batch_indx:batch_indx + batch_size]
        lst_lbl = list_labels[batch_indx:batch_indx + batch_size]

    except IndexError as e:
        print(f"IndexError: {e}\nCheck col_num value: {col_num}\nData shape: {data.shape}")
        lst_data, lst_lbl = None, None  # Return empty batch if error occurs

    return lst_data, lst_lbl
'''
Reading the data
'''
import os
folder_path = "reduced3/"
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

all_data = []

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    print(f"Processing file: {csv_file}")
    
    csv_reader = pd.read_csv(file_path, low_memory=False)
    csv_reader.columns = csv_reader.columns.str.strip()
    csv_reader.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], errors='ignore', inplace=True)
    csv_reader.fillna(0, inplace=True)
    
    all_data.append(csv_reader)

# Combine all data into a single DataFrame
data_combined = pd.concat(all_data, ignore_index=True)

# Save combined features
features_test = np.array([x.strip() for x in data_combined.columns.to_numpy()])
np.savetxt("Experiments/Features.csv", features_test, delimiter=",", fmt='%s')

data = data_combined.to_numpy()


col_num = data.shape[1]
row_num = data.shape[0]

labels = np.array(data[:, col_num - 1])
labels, unique_labels = pd.factorize(labels)

print("Labels converted successfully!")
print("Label Mapping:", dict(enumerate(unique_labels)))
#######labels = labels.astype(int)
########print(np.unique(labels))
# dropping the labels' columns
data = data[:, :col_num - 1]
''' 1-d array to one-hot conversion
# Labels 1 >>> 1 0 >>> Positive
# Labels 2 >>> 0 1 >>> Negative
# In the rules, output 0 means (1 0)
# In the rules, output 1 means (0 1)'''

onehot_labels = np.zeros((row_num, N_CLASSSES))
onehot_labels[np.arange(labels.size), labels - 1] = 1

'''
Computation Graph
'''
X = tf.placeholder(shape=[None, input], dtype=tf.float32, name="X")
y = tf.placeholder(shape=[None, None], dtype=tf.float32, name="y")

'''
ypred=prediction=The output of the capsnet for class prediction
b_lst=cpl_lst=Routing weights list
a_lst=pred_lst=Caps-predicted list
out_vect=caps_out list
'''

dt, prediction, cpl_lst, pred_lst, out_vect, caps_out, W_tiled = neural_network_model(X)

'''Reconstruction Loss'''
mask_with_labels = tf.placeholder_with_default(True, shape=(), name="mask_with_labels")
decoder_inp = reconstruction(y, prediction, caps_out, mask_with_labels)
decoder_out = decoder(decoder_inp)

X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_out,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                    name="reconstruction_loss")

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5
caps_out_norm = safe_norm(caps_out, axis=-2, keepdims=True,
                              name="caps_out_norm")
present_error_raw = tf.square(tf.maximum(0., m_plus - caps_out_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, size_layers[-1]),
                           name="present_error")
absent_error_raw = tf.square(tf.maximum(0., caps_out_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1,  size_layers[-1]),
                          name="absent_error")
L = tf.add(y * present_error, lambda_ * (1.0 - y) * absent_error,
           name="L")
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
'''Final Loss'''
eta = 5.0e-15
cost = tf.add(margin_loss, eta * reconstruction_loss, name="loss")

correct = tf.equal(tf.argmax(y, -1), tf.argmax(prediction, -1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")


# --------------------------------------------------------------------------
# Optimize
# --------------------------------------------------------------------------
optimizer = tf.train.AdamOptimizer(learning_rate=LRATE).minimize(cost)
# --------------------------------------------------------------------------

init = tf.global_variables_initializer()
saver = tf.train.Saver()

