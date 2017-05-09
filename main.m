%% Initialization
clear ; close all; clc


%% Setup the Default parameters for this MLP
input_layer_size  = 784;  % 28x28 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (Note: mapped "0" to label 10)
nhidden_layers = 2;
nnodes = [50 50];
%% 1 -> ReLU,  2 -> Tanh,  3 -> Sigmoid

actfun = [1 1 1];
total_layers = nhidden_layers + 2;
nodes_per_layer = [input_layer_size nnodes num_labels];

%%===================== Take parameters ========================
% initialize the MLP
[nhidden_layers, nnodes, actfun] = my_mlp_init();
total_layers = nhidden_layers + 2;

%pause;

%% =========== Loading and Visualizing Data =============
% Load Training Data
fprintf('Loading Data ...\n')
X_train = loadMNISTImages("train-images-idx3-ubyte");
X_train = X_train';                                       % 50,000 x 784
y_train = loadMNISTLabels("train-labels-idx1-ubyte");
y_train(find(y_train == 0)) = 10;                         % map 0 to 10

% Extract validation data
select_rows = randperm(size(X_train, 1), 10000);                            % randomly
X_valid = X_train(select_rows, :);
X_train(select_rows, :) = [];
y_valid = y_train(select_rows);
y_train(select_rows) = [];

% Load Testing Data
X_test = loadMNISTImages("t10k-images.idx3-ubyte");
X_test = X_test';                                               % 10,000 x 784
y_test = loadMNISTLabels("t10k-labels.idx1-ubyte");
y_test(find(y_test == 0)) = 10;

m = size(X_train, 1);                                           % 60,000
input_layer_size = size(X_train, 2);                            % 784

nodes_per_layer = [input_layer_size nnodes num_labels];

fprintf('\nMulti Layer perceptron design:\n');
fprintf('Total layers: %d, Number of hidden layers: %d\n', total_layers, nhidden_layers);
fprintf('Nodes per layer: ');
disp(nodes_per_layer);



% Randomly select 100 data points to display
%sel = randperm(size(X_train, 1));
%sel = sel(1:100);

%displayData(X_train(sel, :));

%% ================ Initializing Parameters ================
% Randomly initialized weights stored in a initial_weights vector
% in unrolled format

fprintf('\nInitializing Neural Network Parameters ...\n')
initial_weights = [];
for L=1:(total_layers - 1)
  initial_weights = [initial_weights ; randInitializeWeights(nodes_per_layer(L), nodes_per_layer(L+1))(:)];
end

%% ================ Compute Cost (Feedforward) ================

fprintf('\nFeedforward MLP ...\n')

lambda = 0;

[Cost grad] = mlp_costAndGrad(initial_weights, total_layers, nodes_per_layer, ...
                    X_train, y_train, lambda, actfun);

fprintf(['Initial Cost : %f \n'], Cost);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== check MLP gradients ================================
%%
if yes_or_no("Check MLP gradients: ")
  lambda = 1;
  if yes_or_no(["Check gradients on small neural network\n( Your designed network"...
                  "may take some time to calculate if weights are too many): "])

    checkNNGradients2(lambda);
  else
    fprintf('\nChecking Backpropagation on first 100 Training data.... \n')

    %  Check gradients by running checkNNGradients
    checkMLPGradients(initial_weights, total_layers, nodes_per_layer, ...
                        X_train(1:100, :), y_train(1:100), lambda, actfun);
  endif
  fprintf('Program paused. Press enter to continue.\n');
  pause;
endif
%%========================== Train MLP =================================

fprintf('Training MLP... \n');

alpha = 0.01;            % learning_rate
lambda = 0;               % regularization parameter
final_weights = mlp_train(initial_weights, total_layers, nodes_per_layer, ...
                            X_train, y_train, alpha, lambda, actfun, "adaGrad", X_valid, y_valid);

fprintf('Training Complete ! \n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%%========================== Final Cost =================================

fprintf('\nFeedforward MLP ...\n')

lambda = 0;

[Cost grad] = mlp_costAndGrad(final_weights, total_layers, nodes_per_layer, ...
                    X_train, y_train, lambda, actfun);

fprintf(['Final Cost of training data : %f \n'], Cost);

[Cost grad] = mlp_costAndGrad(final_weights, total_layers, nodes_per_layer, ...
                    X_test, y_test, lambda, actfun);

fprintf(['Final Cost of Test data : %f \n'], Cost);


%%========================== Predicting on Training set ======================

fprintf("Predictions on Training set\n");
predictions = mlp_predict(final_weights, X_train, total_layers, nodes_per_layer, actfun);
y_train(find(y_train == 10)) = 0;
fprintf('Training Set Accuracy: %f\n', mean(double(predictions == y_train)) * 100);

%%========================== Predicting on Validation set ======================

fprintf("Predictions on Validation set\n");
predictions = mlp_predict(final_weights, X_valid, total_layers, nodes_per_layer, actfun);
y_valid(find(y_valid == 10)) = 0;
fprintf('\nValidation Set Accuracy: %f\n', mean(double(predictions == y_valid)) * 100);

%%========================== Predicting on Test set ======================

fprintf("Predictions on test set\n");
y_test(find(y_test == 10)) = 0;

predictions = mlp_predict(final_weights, X_test, total_layers, nodes_per_layer, actfun, true);

fprintf('\nTest Set Accuracy: %f\n', mean(double(predictions == y_test)) * 100);
