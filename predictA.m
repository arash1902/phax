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
function p = predict(Theta1, Theta1n, Theta2, Theta2n, Theta2_1,  Theta2_1n,...
                Theta2_2, Theta2_2n,...
                X, num_labels, sharp_factor, sharp_factorn,...
                num_hidden_layer,K , Vdd)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT outputs the predicted label of X given the
%   trained sigmal values of a neural network but it is analog version 
%   since it use sigmal values.

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), num_labels);

X1 = [((Vdd / 2)*ones(m, 1)) X];
X2 = [(-(Vdd / 2)*ones(m, 1)) -X];
weights1 = weight_mapping_wosig([Theta1 Theta1n],K(1));
 
sigma1 = (X1 * weights1(1:size(Theta1,1),1:size(Theta1,2))') +...
        (X2 * weights1(1:size(Theta1,1),(size(Theta1,2)+1):end)') ;
h1 = sigmoid(sigma1,sharp_factor,Vdd);
if (num_hidden_layer == 2)
    weights2_1 = weight_mapping_wosig([Theta2_1 Theta2_1n],K(2));
    weights2 = weight_mapping_wosig([Theta2 Theta2n],K(4));
    sigma2 = [((Vdd / 2)*ones(m, 1)) h1] * weights2_1(1:size(Theta2_1,1),1:size(Theta2_1,2))' + ...
        [(-(Vdd / 2)*ones(m, 1)) sigmoidn(sigma1,sharp_factorn,Vdd)] * ...
        weights2_1(1:size(Theta2_1,1),(size(Theta2_1,2)+1):end)';
    h2 = sigmoid(sigma2,sharp_factor,Vdd);
    sigmaO = [((Vdd / 2)*ones(m, 1)) h2] * weights2(1:size(Theta2,1),1:size(Theta2,2))' + ...
        [(-(Vdd / 2)*ones(m, 1)) sigmoidn(sigma2,sharp_factorn,Vdd)] * ...
        weights2(1:size(Theta2,1),(size(Theta2,2)+1):end)';
    p = sigmoidOut(sigmaO,sharp_factor,Vdd);
elseif(num_hidden_layer == 3)
    weights2_1 = weight_mapping_wosig([Theta2_1 Theta2_1n],K(2));
    weights2_2 = weight_mapping_wosig([Theta2_2 Theta2_2n],K(3));
    weights2 = weight_mapping_wosig([Theta2 Theta2n],K(4));
    sigma2 = [((Vdd / 2)*ones(m, 1)) h1] * weights2_1(1:size(Theta2_1,1),1:size(Theta2_1,2))' + ...
        [(-(Vdd / 2)*ones(m, 1)) sigmoidn(sigma1,sharp_factorn,Vdd)] * ...
        weights2_1(1:size(Theta2_1,1),(size(Theta2_1,2)+1):end)';
    h2 = sigmoid(sigma2,sharp_factor,Vdd);
    sigma3 = [((Vdd / 2)*ones(m, 1)) h2] * weights2_2(1:size(Theta2_2,1),1:size(Theta2_2,2))' + ...
        [(-(Vdd / 2)*ones(m, 1)) sigmoidn(sigma2,sharp_factorn,Vdd)] *...
        weights2_2(1:size(Theta2_2,1),(size(Theta2_2,2)+1):end)';
    h3 = sigmoid(sigma3,sharp_factor,Vdd);
    sigmaO = [((Vdd / 2)*ones(m, 1)) h3] * weights2(1:size(Theta2,1),1:size(Theta2,2))' + ...
        [(-(Vdd / 2)*ones(m, 1)) sigmoidn(sigma3,sharp_factorn,Vdd)] * ...
        weights2(1:size(Theta2,1),(size(Theta2,2)+1):end)';
    p = sigmoidOut(sigmaO,sharp_factor,Vdd);
else
    weights2 = weight_mapping_wosig([Theta2 Theta2n],K(4));
    sigmaO = [((Vdd / 2)*ones(m, 1)) h1] * weights2(1:size(Theta2,1),1:size(Theta2,2))' + ...
        [(-(Vdd / 2)*ones(m, 1)) sigmoidn(sigma1,sharp_factorn,Vdd)] * ...
        weights2(1:size(Theta2,1),(size(Theta2,2)+1):end)';
    p = sigmoidOut(sigmaO,sharp_factor,Vdd);
end
% [dummy, p] = max(h2, [], 2);

% =========================================================================


end
