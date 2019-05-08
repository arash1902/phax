% ============================================================================%
%                 PHAX- initialize weights and do backprop                   %
%                                                                            %
%                      Arash Fayyazi and Mohammad Ansari                     %
%          Low-Power High-Performance Nanosystems Laboratory, Tehran         %
%     SPORT Lab, University of Southern California, Los Angeles, CA 90089    %
%                          http://nanolab.ut.ac.ir/                          %
%                          http://sportlab.usc.edu/                          %
%                                                                            %
%   These models may be freely copied and used for research purposes under   %
%                         the BSD 3-Clause License.                          %
%                                                                            %
%                                                                            %
% Please cite following paper:                                               %
% M. Ansari et al., "PHAX: Physical Characteristics AwareEx-SituTraining     %
% Framework for Inverter-Based Memristive Neuromorphic Circuits," in IEEE    %
% Transactions on Computer-Aided Design of Integrated Circuits and Systems,  %
% vol. 37, no. 8, pp. 1602-1613, Aug. 2018. doi: 10.1109/TCAD.2017.2764070   %
%                                                                            %
%                                                                            %
% ============================================================================%
function [Theta1, Theta1n, Theta2, Theta2n, Theta2_1, Theta2_1n,...
    Theta2_2, Theta2_2n, Final_cost, stop_sign] =  ...
                              my_backpropagation(Vdd, K, num_hidden_layer,...
                                input_layer_size, hidden_layer_size, hidden_layer_size2,...
                                hidden_layer_size3, num_labels, ...
                                X, y,sharp_factor, sharp_factorn, max_iter, lambda,...
                                do_initial, Theta1_init, Theta1n_init, Theta2_init,...
                                Theta2n_init, Theta2_1_init, Theta2_1n_init, ...
                                Theta2_2_init, Theta2_2n_init)

%%% Initialization%%%                            
if ~exist('do_initial', 'var') || isempty(do_initial)
    do_initial = 1;
end
if ~exist('Theta1_init', 'var') || isempty(Theta1_init)
    Theta1_init = [];
end
if ~exist('Theta2_init', 'var') || isempty(Theta2_init)
    Theta2_init = [];
end
if ~exist('Theta2_1_init', 'var') || isempty(Theta2_1_init)
    Theta2_1_init = [];
end

if ~exist('Theta2_2_init', 'var') || isempty(Theta2_2_init)
    Theta2_2_init = [];
end

if ~exist('Theta1n_init', 'var') || isempty(Theta1n_init)
    Theta1n_init = [];
end
if ~exist('Theta2n_init', 'var') || isempty(Theta2n_init)
    Theta2n_init = [];
end
if ~exist('Theta2_1n_init', 'var') || isempty(Theta2_1n_init)
    Theta2_1n_init = [];
end

if ~exist('Theta2_2n_init', 'var') || isempty(Theta2_2n_init)
    Theta2_2n_init = [];
end

Theta2_1 = [];
Theta2_1n = [];
Theta2_2 = [];
Theta2_2n = [];
 
%%% end Initialization  %%%
% %% Setup the parameters you will use for this exercise
% input_layer_size  = 400;  % 20x20 Input Images of Digits
% hidden_layer_size = 25;   % 25 hidden units
% num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

stop_sign = 0;
% load('ex4data1.mat');
X = X';
y = y';
m = size(X, 1);
%%%for weights constraind %%%
Ron = 100;
Roff = 16e3;
rat = Roff./Ron;
%%% finish
if (do_initial)
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta1n = randInitializeWeights(input_layer_size, hidden_layer_size);
    if (num_hidden_layer == 2)
        initial_Theta2_1 = randInitializeWeights(hidden_layer_size, hidden_layer_size2);
        initial_Theta2_1n = randInitializeWeights(hidden_layer_size, hidden_layer_size2);
        initial_Theta2 = randInitializeWeights(hidden_layer_size2, num_labels);
        initial_Theta2n = randInitializeWeights(hidden_layer_size2, num_labels);
        initial_nn_params = [initial_Theta1(:) ; initial_Theta1n(:) ; ...
            initial_Theta2_1(:) ; initial_Theta2_1n(:) ; ...
            initial_Theta2(:) ; initial_Theta2n(:)];
    elseif (num_hidden_layer == 3)
        initial_Theta2_1 = randInitializeWeights(hidden_layer_size, hidden_layer_size2);
        initial_Theta2_1n = randInitializeWeights(hidden_layer_size, hidden_layer_size2);
        initial_Theta2_2 = randInitializeWeights(hidden_layer_size2, hidden_layer_size3);
        initial_Theta2_2n = randInitializeWeights(hidden_layer_size2, hidden_layer_size3);
        initial_Theta2 = randInitializeWeights(hidden_layer_size3, num_labels);
        initial_Theta2n = randInitializeWeights(hidden_layer_size3, num_labels);
        initial_nn_params = [initial_Theta1(:) ; initial_Theta1n(:) ; ...
            initial_Theta2_1(:) ;  initial_Theta2_1n(:) ; ...
            initial_Theta2_2(:) ; initial_Theta2_2n(:) ; ...
            initial_Theta2(:) ; initial_Theta2n(:)];
    else
        initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
        initial_Theta2n = randInitializeWeights(hidden_layer_size, num_labels);
        initial_nn_params = [initial_Theta1(:) ; initial_Theta1n(:) ; ...
            initial_Theta2(:) ; initial_Theta2n(:)]; 
    end
else
     initial_nn_params = [Theta1_init(:) ; Theta1n_init(:) ; Theta2_1_init(:) ;...
                            Theta2_1n_init(:) ;Theta2_2_init(:) ; Theta2_2n_init(:) ;...
                            Theta2_init(:) ; Theta2n_init(:)];
end


% Unroll parameters

fprintf('\nTraining Neural Network... \n')
%  change the MaxIter to a larger
%  value to see how more training helps.

options = optimset('MaxIter', max_iter);
%options = optimset('GradObj', 'on', 'MaxIter', 50);
%  You should also try different values of lambda

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction2(p, ...
                                   num_hidden_layer, ...
                                   input_layer_size, ...
                                   hidden_layer_size, hidden_layer_size2, ...
                                   hidden_layer_size3, num_labels, ...
                                   X, y, lambda,...
                                   sharp_factor, sharp_factorn, K, Vdd);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% last cost function
Final_cost = cost(end);
% Obtain Theta1 and Theta2 and Theta2_1 and Theta2_2 back from nn_params %%
size_h = 0;
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta1n = reshape(nn_params((1+(hidden_layer_size * (input_layer_size + 1))) :...
                 2 * hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));            
if (num_hidden_layer == 2)
    size_h = (2*(hidden_layer_size * (input_layer_size + 1))) + ...
                (hidden_layer_size2 * (hidden_layer_size + 1));
    size_hn = (2*(hidden_layer_size * (input_layer_size + 1))) + ...
                2*(hidden_layer_size2 * (hidden_layer_size + 1));
    size_O = size_hn + (num_labels * (hidden_layer_size2 + 1)) ;
    Theta2_1 = reshape(nn_params(1+(2*hidden_layer_size * (input_layer_size + 1)): ...
                 size_h), hidden_layer_size2, (hidden_layer_size + 1));
    Theta2_1n = reshape(nn_params(1+size_h: ...
                 size_hn), hidden_layer_size2, (hidden_layer_size + 1));
    Theta2 = reshape(nn_params((1 + size_hn):size_O), ...
                 num_labels, (hidden_layer_size2 + 1));
    Theta2n = reshape(nn_params((1 + size_O):end), ...
                 num_labels, (hidden_layer_size2 + 1));
    
elseif (num_hidden_layer == 3)
    size_h = (2*(hidden_layer_size * (input_layer_size + 1))) + ...
                (hidden_layer_size2 * (hidden_layer_size + 1));
    size_hn = (2*(hidden_layer_size * (input_layer_size + 1))) + ...
                2*(hidden_layer_size2 * (hidden_layer_size + 1));
    size_h2 = size_hn + (hidden_layer_size3 * (hidden_layer_size2 + 1));
    size_hn2 = size_h2 + (hidden_layer_size3 * (hidden_layer_size2 + 1));
    size_O = size_hn2 + (num_labels * (hidden_layer_size3 + 1)) ;
    Theta2_1 = reshape(nn_params(1+(2*hidden_layer_size * (input_layer_size + 1)): ...
                 size_h), hidden_layer_size2, (hidden_layer_size + 1));
    Theta2_1n = reshape(nn_params(1+size_h: ...
                 size_hn), hidden_layer_size2, (hidden_layer_size + 1));
    Theta2_2 = reshape(nn_params((size_hn+1) : size_h2), ...
                 hidden_layer_size3, (hidden_layer_size2 + 1));
    Theta2_2n = reshape(nn_params((size_h2+1) : size_hn2), ...
                 hidden_layer_size3, (hidden_layer_size2 + 1));
    Theta2 = reshape(nn_params((size_hn2 + 1) :size_O), ...
                 num_labels, (hidden_layer_size3 + 1));
    Theta2n = reshape(nn_params((size_O + 1) :end), ...
                 num_labels, (hidden_layer_size3 + 1));
    
else
    size_O = (2*hidden_layer_size * (input_layer_size + 1)) + ...
                (num_labels * (hidden_layer_size + 1)) ;
    Theta2 = reshape(nn_params((1 + (2*hidden_layer_size * (input_layer_size + 1))):size_O), ...
                 num_labels, (hidden_layer_size + 1));
    Theta2n = reshape(nn_params((1 + size_O):end), ...
                 num_labels, (hidden_layer_size + 1));
end             

%% predict using the hardware implementation 
pred = predict(Theta1, Theta1n, Theta2, Theta2n, Theta2_1,  Theta2_1n,...
                Theta2_2, Theta2_2n,...
                X, num_labels, sharp_factor, sharp_factorn,...
                num_hidden_layer,K, Vdd);

%stop_sign =mean(mean(double(round(pred*10) == (y*10)))) * 100;
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


