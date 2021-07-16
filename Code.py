# Author: Muhammed Rahmetullah Kartal
# This code implements a one hidden layered neural network in given data

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# normalizing a column with min max normalization
def normalize3(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

# activation function
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# for the loss calculation
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# activation function for output layers
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# splitting one feature into multiple columns in numpy array
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, int(Y.max() + 1)))
    one_hot_Y[np.arange(Y.size), Y.astype(int)] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# weights are calculated with Xavier formula
def create_network(I_dim, H_dim, O_dim):
    weights_ItoH = np.random.randn((H_dim), (I_dim)) * np.sqrt(2 / ((I_dim) + (H_dim)))
    biasH = np.random.rand(H_dim, 1) - 0.5

    weights_HtoO = np.random.randn((O_dim), (H_dim)) * np.sqrt(2 / ((H_dim) + (O_dim)))
    biasO = np.random.rand(O_dim, 1) - 0.5
    return weights_ItoH, biasH, weights_HtoO, biasO


# feedforward propagation, updating weights with given X values and bias term
def forward_prop(weights_ItoH, biasH, weights_HtoO, biasO, X):
    preActivation_H = np.dot(weights_ItoH, X) + biasH
    postActivation_H = sigmoid(preActivation_H)

    preActivation_O = np.dot(weights_HtoO, postActivation_H) + biasO
    postActivation_O = softmax(preActivation_O)

    return preActivation_H, postActivation_H, preActivation_O, postActivation_O


# re updating weights and bias related to loss function
def backward_prop(preActivation_H, postActivation_H, postActivation_O, weights_HtoO, X, Y):
    dSize = len(Y)
    # we are using softmax function for output layer so we used one hot encoding to calculate error
    one_hot_Y = one_hot(Y)
    der_preActivation_O = postActivation_O - one_hot_Y
    der_weights_HtoO = (1 / dSize) * np.dot(der_preActivation_O, postActivation_H.T)
    der_biasO = (1 / dSize) * np.sum(der_preActivation_O)

    # used sigmoid derivative to calculate error rate
    der_preActivation_H = np.dot(weights_HtoO.T, der_preActivation_O) * sigmoid_derivative(preActivation_H)
    der_weights_ItoH = (1 / dSize) * np.dot(der_preActivation_H, X.T)
    der_biasH = (1 / dSize) * np.sum(der_preActivation_H)

    return der_weights_ItoH, der_biasH, der_weights_HtoO, der_biasO

# updating weights and bias values
def update_network(weights_ItoH, biasH, weights_HtoO, biasO, der_weights_ItoH, der_biasH, der_weights_HtoO, der_biasO,
                   LR):
    weights_ItoH = weights_ItoH - LR * der_weights_ItoH
    biasH -= LR * der_biasH
    weights_HtoO = weights_HtoO - LR * der_weights_HtoO
    biasO -= LR * der_biasO
    return weights_ItoH, biasH, weights_HtoO, biasO


# shuffling splitting data into n fold, taking the first 1/n as test, rest as train
def cross_val(data, n_fold):
    data = np.array(data)
    np.random.shuffle(data)

    fold_size = int(data.shape[0] / n_fold)

    data_val = data[0:fold_size].T
    y_val = data_val[-1]
    x_val = data_val[0:(data.shape[1] - 1)]

    data_train = data[fold_size:data.shape[0]].T
    y_train = data_train[-1]
    x_train = data_train[0:(data.shape[1] - 1)]

    return x_val, y_val, x_train, y_train


# accuracy calculator
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def get_loss(f_v, y):
    y = one_hot(y)
    return -np.sum([y[i]*np.log2(f_v[i]) for i in range(len(y))])


# make predictions with re applying forward propagation
def make_predictions(X, weights_ItoH, biasH, weights_HtoO, biasO):
    _, _, _, postActivation_O = forward_prop(weights_ItoH, biasH, weights_HtoO, biasO, X)
    predictions = np.argmax(postActivation_O, 0)
    return predictions

def make_eval(X, weights_ItoH, biasH, weights_HtoO, biasO):
    _, _, _, postActivation_O = forward_prop(weights_ItoH, biasH, weights_HtoO, biasO, X)
    return postActivation_O


def gradient_descent(X, Y, LR, n_epochs, H_size, X_val, Y_val):
    I_size = X.shape[0]
    O_size = len(np.unique(Y))
    accuracy, accuracy_v = 0, 0

    graphStore, graphStore_v = pd.DataFrame(), pd.DataFrame()

    weights_ItoH, biasH, weights_HtoO, biasO = create_network(I_size, H_size, O_size)
    for i in range(n_epochs):
        if accuracy == 1 or accuracy_v == 1:
            break
        for phase in ['train', 'val']:
            if phase == 'train':
                preActivation_H, postActivation_H, preActivation_O, postActivation_O = forward_prop(weights_ItoH, biasH,
                                                                                                    weights_HtoO, biasO,
                                                                                                    X)
                der_weightsItoH, der_biasH, der_weightsHtoO, der_biasO = backward_prop(preActivation_H,
                                                                                       postActivation_H,
                                                                                       postActivation_O, weights_HtoO,
                                                                                       X, Y)
                weights_ItoH, biasH, weights_HtoO, biasO = update_network(weights_ItoH, biasH, weights_HtoO, biasO,
                                                                          der_weightsItoH, der_biasH, der_weightsHtoO,
                                                                          der_biasO, LR)

                if i % 10 == 0:
                    prediction = np.argmax(postActivation_O, 0)
                    accuracy = get_accuracy(prediction, Y)
                    loss = get_loss(postActivation_O, Y) / (Y.shape[0])

                    x = {"Iteration": i, "Accuracy": accuracy, "Loss": loss}
                    x = pd.DataFrame([x])
                    graphStore = pd.concat([graphStore, x])
                    print(x)

            elif phase == 'val':
                if i % 10 == 0:
                    preds = make_eval(X_val, weights_ItoH, biasH, weights_HtoO, biasO)

                    accuracy_v = get_accuracy(np.argmax(preds, 0), Y_val)
                    loss_v = get_loss(preds, Y_val) / (Y_val.shape[0])

                    x_v = {"Iteration": i, "Accuracy": accuracy_v, "Loss": loss_v}
                    x_v = pd.DataFrame([x_v])
                    graphStore_v = pd.concat([graphStore_v, x_v])
                    # print("Val",x_v)

    return weights_ItoH, biasH, weights_HtoO, biasO, graphStore, graphStore_v

#preparing data
data = pd.read_csv("/content/drive/Shareddrives/MLTermProject/healthcare-dataset-stroke-data.csv")
data = data.drop(['id'],axis=1)
data["bmi"].replace(np.NaN,data["bmi"].mean(),inplace=True)
data["bmi"] = normalize3(data["bmi"])
data["avg_glucose_level"] = normalize3(data["avg_glucose_level"])
data["age"] = normalize3(data["age"])
data = pd.get_dummies(data, prefix=['gender', 'ever_married','work_type','Residence_type','smoking_status'])


x_test, y_test, x_train, y_train= cross_val(data,8)
data = pd.concat([pd.DataFrame(x_train.T),pd.DataFrame(y_train.T)],axis=1)
x_val, y_val, x_train, y_train= cross_val(data,6)

weights_ItoH, biasH, weights_HtoO, biasO, graphStore, graphStore_v = gradient_descent(x_train, y_train, 0.1, 2000, 20,x_val,y_val)

preds = make_predictions(x_val, weights_ItoH, biasH, weights_HtoO, biasO)
print("Prediction Accuracy: ",get_accuracy(preds, y_val))
print("Prediction F1:", f1_score(y_val, preds, average='macro'))

sns.lineplot(data = graphStore, x='Iteration', y='Accuracy',label="Train")
sns.lineplot(data = graphStore_v, x='Iteration', y='Accuracy',label = "Validation")
plt.show()
sns.lineplot(data = graphStore, x='Iteration', y='Loss',label="Train")
sns.lineplot(data = graphStore_v, x='Iteration', y='Loss',label = "Validation")
plt.show()




#trying different parameters
LRS = [0.001,0.01,0.1,0.5]
LAYERS = [2,5,8,15,20,25,50]
for lay in LAYERS:
  for lr in LRS:
    weights_ItoH, biasH, weights_HtoO, biasO, graphStore, graphStore_v = gradient_descent(x_train, y_train, lr, 2000, lay ,x_val,y_val)
    preds = make_predictions(x_val, weights_ItoH, biasH, weights_HtoO, biasO)
    print("LR:", lr, "Layer:", lay)
    print("Prediction Accuracy: ",get_accuracy(preds, y_val))
    print("Prediction F1:", f1_score(y_val, preds, average='macro'))

    sns.lineplot(data = graphStore, x='Iteration', y='Accuracy',label="Train")
    sns.lineplot(data = graphStore_v, x='Iteration', y='Accuracy',label = "Validation")
    plt.show()
    sns.lineplot(data = graphStore, x='Iteration', y='Loss',label="Train")
    sns.lineplot(data = graphStore_v, x='Iteration', y='Loss',label = "Validation")
    plt.show()




# data = pd.read_csv("/content/drive/Shareddrives/MLTermProject/healthcare-dataset-stroke-data.csv")
# data = data.drop(['id'],axis=1)
# data["bmi"].replace(np.NaN,data["bmi"].mean(),inplace=True)
# data["bmi"] = normalize3(data["bmi"])
# data["avg_glucose_level"] = normalize3(data["avg_glucose_level"])
# data["age"] = normalize3(data["age"])
# data = pd.get_dummies(data, prefix=['gender', 'ever_married','work_type','Residence_type','smoking_status'])
# x_val, y_val, x_train, y_train = cross_val(data,10)

# data = pd.read_csv('/content/drive/Shareddrives/MLTermProject/Iris.csv')
# data = data.drop(['Id'],axis=1)
# data = pd.get_dummies(data, prefix=['Species'])
# x_val, y_val, x_train, y_train = cross_val(data,10)



#sklearn
x_train = x_train.T
x_val = x_val.T

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='sgd', alpha=0.1, hidden_layer_sizes=(20), random_state=1)

N_EPOCHS = 20
epoch = 0
N_BATCH = 15
N_CLASSES = np.unique(y_train)
N_TRAIN_SAMPLES = x_train.shape[0]

scores_train = []
scores_test = []

while epoch < N_EPOCHS:
    # SHUFFLING
    random_perm = np.random.permutation(x_train.shape[0])
    mini_batch_index = 0
    while True:
        # MINI-BATCH
        indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
        mlp.partial_fit(x_train[indices], y_train[indices], classes=N_CLASSES)
        mini_batch_index += N_BATCH

        if mini_batch_index >= N_TRAIN_SAMPLES:
            break

    # SCORE TRAIN
    scores_train.append(mlp.score(x_train,y_train))
    # SCORE TEST
    scores_test.append(mlp.score(x_val, y_val))
    epoch += 1

fig, ax = plt.subplots(1, sharex=True, sharey=True)
ax.plot(scores_train)
ax.plot(scores_test)
fig.suptitle("Accuracy over epochs", fontsize=14)
plt.show()



#Iris dataset
data = pd.read_csv('/content/drive/Shareddrives/MLTermProject/Iris.csv')
data = data.drop(['Id'],axis=1)
data = pd.get_dummies(data, prefix=['Species'])
x_val, y_val, x_train, y_train = cross_val(data,10)
weights_ItoH, biasH, weights_HtoO, biasO, graphStore,graphStore_v= gradient_descent(x_train, y_train, 0.1, 2000, 20,x_val,y_val)

preds = make_predictions(x_val, weights_ItoH, biasH, weights_HtoO, biasO)#test
print("Prediction Accuracy: ",get_accuracy(preds, y_val))
print("Prediction F1:", f1_score(y_val, preds, average='macro'))
sns.lineplot(data = graphStore, x='Iteration', y='Accuracy',label="Train")
sns.lineplot(data = graphStore_v, x='Iteration', y='Accuracy',label = "Validation")
plt.show()
sns.lineplot(data = graphStore, x='Iteration', y='Loss',label="Train")
sns.lineplot(data = graphStore_v, x='Iteration', y='Loss',label = "Validation")
plt.show()