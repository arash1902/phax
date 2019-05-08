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
function [J grad] = nnCostFunction2(nn_params, ...
                                   num_hidden_layer, ...
                                   input_layer_size, ...       
                                   hidden_layer_size, hidden_layer_size2, ...
                                   hidden_layer_size3, num_labels, ...
                                   X, y, lambda,...
                               sharp_factor,sharp_factorn,K, Vdd)
                           
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, ...
%   computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
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


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta1n_grad = zeros(size(Theta1n));
Theta2n_grad = zeros(size(Theta2n));

if (num_hidden_layer == 2)
    Theta2_1_grad = zeros(size(Theta2_1));
    Theta2_1n_grad = zeros(size(Theta2_1n));
    z_2_1 = zeros(1,(hidden_layer_size+1));
    a_2_1 = zeros(1,(hidden_layer_size+1));
    delta_2_1 = zeros(hidden_layer_size,1);
    Delta_2_1 = zeros(size(Theta2_1));
elseif (num_hidden_layer == 3)
    Theta2_1_grad = zeros(size(Theta2_1));
    Theta2_2_grad = zeros(size(Theta2_2));
    Theta2_1n_grad = zeros(size(Theta2_1n));
    Theta2_2n_grad = zeros(size(Theta2_2n));
    z_2_1 = zeros(1,(hidden_layer_size+1));
    a_2_1 = zeros(1,(hidden_layer_size+1));
    z_2_2 = zeros(1,(hidden_layer_size+1));
    a_2_2 = zeros(1,(hidden_layer_size+1));
    delta_2_1 = zeros(hidden_layer_size,1);
    delta_2_2 = zeros(hidden_layer_size,1);
    Delta_2_1 = zeros(size(Theta2_1));
    Delta_2_2 = zeros(size(Theta2_2));
end
a_1 = zeros(1,(input_layer_size+1));
z_2 = zeros(1,(hidden_layer_size+1));
a_2 = zeros(1,(hidden_layer_size+1));
z_3 = zeros(1,num_labels);
a_3 = zeros(1,num_labels);   
delta_3 = zeros(num_labels,1);
delta_2 = zeros(hidden_layer_size,1);
Delta_2 = zeros(size(Theta2));
Delta_1 = zeros(size(Theta1));


%%%%%%%%% feed forward
X1 = [((Vdd / 2)*ones(m, 1)) X];
X2 = [(-(Vdd / 2)*ones(m, 1)) -X];
weights1 = weight_mapping([Theta1 Theta1n],K(1));
 
sigma1 = (X1 * weights1(1:size(Theta1,1),1:size(Theta1,2))') +...
        (X2 * weights1(1:size(Theta1,1),(size(Theta1,2)+1):end)') ;
h1 = sigmoid(sigma1,sharp_factor,Vdd);
if (num_hidden_layer == 2)
    weights2_1 = weight_mapping([Theta2_1 Theta2_1n],K(2));
    weights2 = weight_mapping([Theta2 Theta2n],K(4));
    sigma2 = [((Vdd / 2)*ones(m, 1)) h1] * weights2_1(1:size(Theta2_1,1),1:size(Theta2_1,2))' + ...
        [(-(Vdd / 2)*ones(m, 1)) sigmoidn(sigma1,sharp_factorn,Vdd)] * ...
        weights2_1(1:size(Theta2_1,1),(size(Theta2_1,2)+1):end)';
    h2 = sigmoid(sigma2,sharp_factor,Vdd);
    sigmaO = [((Vdd / 2)*ones(m, 1)) h2] * weights2(1:size(Theta2,1),1:size(Theta2,2))' + ...
        [(-(Vdd / 2)*ones(m, 1)) sigmoidn(sigma2,sharp_factorn,Vdd)] * ...
        weights2(1:size(Theta2,1),(size(Theta2,2)+1):end)';
    h_theta = sigmoidOut(sigmaO,sharp_factor,Vdd);
elseif(num_hidden_layer == 3)
    weights2_1 = weight_mapping([Theta2_1 Theta2_1n],K(2));
    weights2_2 = weight_mapping([Theta2_2 Theta2_2n],K(3));
    weights2 = weight_mapping([Theta2 Theta2n],K(4));
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
    h_theta = sigmoidOut(sigmaO,sharp_factor,Vdd);
else
    weights2 = weight_mapping([Theta2 Theta2n],K(4));
    sigmaO = [((Vdd / 2)*ones(m, 1)) h1] * weights2(1:size(Theta2,1),1:size(Theta2,2))' + ...
        [(-(Vdd / 2)*ones(m, 1)) sigmoidn(sigma1,sharp_factorn,Vdd)] * ...
        weights2(1:size(Theta2,1),(size(Theta2,2)+1):end)';
    h_theta = sigmoidOut(sigmaO,sharp_factor,Vdd);
end

% %%%
%%%%%%%%% Cost calculation
h_theta = (1 / Vdd) .* (h_theta+(Vdd/2));
h_theta_pos =(h_theta == 0) * realmin + h_theta;
h_theta_neg = 1-h_theta;
h_theta_neg =(h_theta_neg == 0) * realmin + h_theta_neg;
J = J + (1/m)*sum(sum( (1 / Vdd) .* ( -((Vdd/2)+y).*log(h_theta_pos) + (-(Vdd/2)+y).*log(h_theta_neg) ) ) );%+ (lambda/(2*m))*(sum(theta.^2) - theta(1)^2);

%%%%
%J = (1/m)*sum(sum( (h_theta - y).^2 ) ) ;
J = J + (lambda/(2*m))*(sum(sum(weight_mapping(Theta1,K(1)).^2)) -...
    sum(sum(weight_mapping(Theta1(:,1),K(1)).^2)) + ...
    sum(sum(weight_mapping(Theta2,K(4)).^2)) -...
    sum(sum(weight_mapping(Theta2(:,1),K(4)).^2)));


%%%%%%%%% Back propagation
    a_1 = [((Vdd / 2) * ones(1, m)); X'];
    a_1n = [(-(Vdd / 2) * ones(1, m)); -X'];
    weights1 = weight_mapping([Theta1 Theta1n],K(1));
    z_2 = weights1(1:size(Theta1,1),1:size(Theta1,2)) * a_1 + ...
        weights1(1:size(Theta1,1),(size(Theta1,2)+1):end) * a_1n;    
    a_2 = [((Vdd / 2)*ones(1, m)); sigmoid(z_2,sharp_factor,Vdd)];
    a_2n = [(-(Vdd / 2)*ones(1, m)); sigmoidn(z_2,sharp_factorn,Vdd)];
    if (num_hidden_layer == 2)
        z_2_1 = weights2_1(1:size(Theta2_1,1),1:size(Theta2_1,2)) * a_2 + ...
        weights2_1(1:size(Theta2_1,1),(size(Theta2_1,2)+1):end) * a_2n;    
        a_2_1 = [((Vdd / 2)*ones(1, m)); sigmoid(z_2_1,sharp_factor,Vdd)];
        a_2_1n = [(-(Vdd / 2)*ones(1, m)); sigmoidn(z_2_1,sharp_factorn,Vdd)];
        z_3 = weights2(1:size(Theta2,1),1:size(Theta2,2)) * a_2_1 + ...
        weights2(1:size(Theta2,1),(size(Theta2,2)+1):end) * a_2_1n;
        a_3 = sigmoidOut(z_3,sharp_factor,Vdd);
        delta_3 = -(a_3 - (y)');
        
        weights2_md = weight_md([Theta2 Theta2n],K(4));
        weights1_md = weight_md([Theta1 Theta1n],K(1));
        weights2_1_md = weight_md([Theta2_1 Theta2_1n],K(4));
        delta_2_1 = (weights2(1:size(Theta2,1),1:size(Theta2,2))' * delta_3) .*...
            [((Vdd / 2)*ones(1, m));(sigmoidGradient(z_2_1,sharp_factor))] + ...
            (weights2(1:size(Theta2,1),(size(Theta2,2)+1):end)' * delta_3) .*...
            [(-(Vdd / 2)*ones(1, m)); sigmoidGradientn(z_2_1,sharp_factorn)];
        delta_2 = (weights2_1(1:size(Theta2_1,1),1:size(Theta2_1,2))' * delta_2_1(2:end, :)) .*...
            [((Vdd / 2)*ones(1, m));(sigmoidGradient(z_2,sharp_factor))]+ ...
            (weights2_1(1:size(Theta2_1,1),(size(Theta2_1,2)+1):end)' * delta_2_1(2:end, :)) .*...
            [(-(Vdd / 2)*ones(1, m)); sigmoidGradientn(z_2,sharp_factorn)];        
        Delta_2t = delta_3 * ([a_2_1;a_2_1n]' * weights2_md);
        for i=1:size(delta_3,1)
            Delta_2(i,:) = Delta_2t(i,((i-1)*2*size(Theta2,2) +1):((2*i - 1)*size(Theta2,2)));
            Delta_2n(i,:) = Delta_2t(i,((2*i - 1)*size(Theta2,2) +1):(2*i*size(Theta2,2)));
        end
        Delta_2_1t = delta_2_1(2:end, :) * ([a_2;a_2n]' * weights2_1_md);
        for i=1:(size(delta_2_1,1)-1)
            Delta_2_1(i,:) = Delta_2_1t(i,((i-1)*2*size(Theta2_1,2) +1):((2*i -1)*size(Theta2_1,2)));
            Delta_2_1n(i,:) = Delta_2_1t(i,((2*i -1)*size(Theta2_1,2) +1):(i*2*size(Theta2_1,2)));
        end
        Delta_1t = delta_2(2:end, :) * ([a_1;a_1n]' * weights1_md);
        for i=1:(size(delta_2,1)-1)
            Delta_1(i,:) = Delta_1t(i,((i-1)*2*size(Theta1,2) +1):((2*i -1)*size(Theta1,2)));
            Delta_1n(i,:) = Delta_1t(i,((2*i -1)*size(Theta1,2) +1):(i*2*size(Theta1,2)));
        end
    elseif (num_hidden_layer == 3)
        z_2_1 = weights2_1(1:size(Theta2_1,1),1:size(Theta2_1,2)) * a_2 + ...
        weights2_1(1:size(Theta2_1,1),(size(Theta2_1,2)+1):end) * a_2n;  
        a_2_1 = [((Vdd / 2)*ones(1, m)); sigmoid(z_2_1,sharp_factor,Vdd)];
        a_2_1n = [(-(Vdd / 2)*ones(1, m)); sigmoidn(z_2_1,sharp_factorn,Vdd)];
        z_2_2 = weights2_2(1:size(Theta2_2,1),1:size(Theta2_2,2)) * a_2_1 + ...
        weights2_2(1:size(Theta2_2,1),(size(Theta2_2,2)+1):end) * a_2_1n;  
        a_2_2 = [((Vdd / 2)*ones(1, m)); sigmoid(z_2_2,sharp_factor,Vdd)];
        a_2_2n = [(-(Vdd / 2)*ones(1, m)); sigmoidn(z_2_2,sharp_factorn,Vdd)];
        z_3 = weights2(1:size(Theta2,1),1:size(Theta2,2)) * a_2_2 + ...
        weights2(1:size(Theta2,1),(size(Theta2,2)+1):end) * a_2_2n;  
        a_3 = sigmoidOut(z_3,sharp_factor,Vdd);      
        delta_3 = -(a_3 - (y)');
        
        weights2_md = weight_md([Theta2 Theta2n],K(4));
        weights1_md = weight_md([Theta1 Theta1n],K(1));
        weights2_1_md = weight_md([Theta2_1 Theta2_1n],K(2));
        weights2_2_md = weight_md([Theta2_2 Theta2_2n],K(3));
        delta_2_2 = (weights2(1:size(Theta2,1),1:size(Theta2,2))' * delta_3) .*...
            [((Vdd / 2)*ones(1, m));(sigmoidGradient(z_2_2,sharp_factor))] + ...
            (weights2(1:size(Theta2,1),(size(Theta2,2)+1):end)' * delta_3) .*...
            [(-(Vdd / 2)*ones(1, m)); sigmoidGradientn(z_2_2,sharp_factorn)];
        delta_2_1 = (weights2_2(1:size(Theta2_2,1),1:size(Theta2_2,2))' * delta_2_2(2:end, :)) .*...
            [((Vdd / 2)*ones(1, m));(sigmoidGradient(z_2_1,sharp_factor))]+ ...
            (weights2_2(1:size(Theta2_2,1),(size(Theta2_2,2)+1):end)' * delta_2_2(2:end, :)) .*...
            [(-(Vdd / 2)*ones(1, m)); sigmoidGradientn(z_2_1,sharp_factorn)];
        delta_2 = (weights2_1(1:size(Theta2_1,1),1:size(Theta2_1,2))' * delta_2_1(2:end, :)) .*...
            [((Vdd / 2)*ones(1, m));(sigmoidGradient(z_2,sharp_factor))] +...
            (weights2_1(1:size(Theta2_1,1),1:size(Theta2_1,2))' * delta_2_1(2:end, :)) .*...
            [(-(Vdd / 2)*ones(1, m)); sigmoidGradientn(z_2,sharp_factorn)];
        Delta_2t = delta_3 * ([a_2_2;a_2_2n]' * weights2_md);
        for i=1:size(delta_3,1)
            Delta_2(i,:) = Delta_2t(i,((i-1)*2*size(Theta2,2) +1):((2*i - 1)*size(Theta2,2)));
            Delta_2n(i,:) = Delta_2t(i,((2*i - 1)*size(Theta2,2) +1):(2*i*size(Theta2,2)));
        end
        Delta_2_2t = delta_2_2(2:end, :) * ([a_2_1;a_2_1n]' * weights2_2_md);
        for i=1:(size(delta_2_2,1)-1)
            Delta_2_2(i,:) = Delta_2_2t(i,((i-1)*2*size(Theta2_2,2) +1):((2*i -1)*size(Theta2_2,2)));
            Delta_2_2n(i,:) = Delta_2_2t(i,((2*i -1)*size(Theta2_2,2) +1):(i*2*size(Theta2_2,2)));
        end
        Delta_2_1t = delta_2_1(2:end, :) * ([a_2;a_2n]' * weights2_1_md);
        for i=1:(size(delta_2_1,1)-1)
            Delta_2_1(i,:) = Delta_2_1t(i,((i-1)*2*size(Theta2_1,2) +1):((2*i -1)*size(Theta2_1,2)));
            Delta_2_1n(i,:) = Delta_2_1t(i,((2*i -1)*size(Theta2_1,2) +1):(i*2*size(Theta2_1,2)));
        end
        Delta_1t = delta_2(2:end, :) * ([a_1;a_1n]' * weights1_md);
        for i=1:(size(delta_2,1)-1)
            Delta_1(i,:) = Delta_1t(i,((i-1)*2*size(Theta1,2) +1):((2*i -1)*size(Theta1,2)));
            Delta_1n(i,:) = Delta_1t(i,((2*i -1)*size(Theta1,2) +1):(i*2*size(Theta1,2)));
        end
    else
        z_3 = weights2(1:size(Theta2,1),1:size(Theta2,2)) * a_2 + ...
                weights2(1:size(Theta2,1),(size(Theta2,2)+1):end) * a_2n;
        a_3 = sigmoidOut(z_3,sharp_factor,Vdd);
        delta_3 = -(a_3 - (y)');
        delta_2 = (weights2(1:size(Theta2,1),1:size(Theta2,2))' * delta_3) .*...
            [((Vdd / 2)*ones(1, m)); sigmoidGradient(z_2,sharp_factor)] + ...
            (weights2(1:size(Theta2,1),(size(Theta2,2)+1):end)' * delta_3) .*...
            [(-(Vdd / 2)*ones(1, m)); sigmoidGradientn(z_2,sharp_factorn)];
        weights2_md = weight_md([Theta2 Theta2n],K(4));
        weights1_md = weight_md([Theta1 Theta1n],K(1));
        Delta_2t = delta_3 * ([a_2;a_2n]' * weights2_md);
        for i=1:size(delta_3,1)
            Delta_2(i,:) = Delta_2t(i,((i-1)*2*size(Theta2,2) +1):((2*i - 1)*size(Theta2,2)));
            Delta_2n(i,:) = Delta_2t(i,((2*i - 1)*size(Theta2,2) +1):(2*i*size(Theta2,2)));
        end
        Delta_1t = delta_2(2:end, :) * ([a_1;a_1n]' * weights1_md);
        for i=1:(size(delta_2,1)-1)
            Delta_1(i,:) = Delta_1t(i,((i-1)*2*size(Theta1,2) +1):((2*i -1)*size(Theta1,2)));
            Delta_1n(i,:) = Delta_1t(i,((2*i -1)*size(Theta1,2) +1):(i*2*size(Theta1,2)));
        end
        
    end
    
%%%%%%%%% gradient calculation
    Theta2_grad(:,1) = (2 / Vdd) * sharp_factor *(1/m) * Delta_2(:,1);
    Theta2_grad(:,2:end) = (2 / Vdd) * sharp_factor * (1/m) * Delta_2(:,2:end) + ...
                            (lambda/m)*Theta2(:,2:end);
    Theta2n_grad(:,1) = (2 / Vdd)* sharp_factor * (1/m) * Delta_2n(:,1);
    Theta2n_grad(:,2:end) = (2 / Vdd) * sharp_factor * (1/m) * Delta_2n(:,2:end) + ...
                            (lambda/m)*Theta2n(:,2:end);
if (num_hidden_layer == 2)
    
    Theta1_grad(:,1) = (Vdd / 2) * sharp_factor *(1/m) * (Delta_1(:,1));
    Theta1_grad(:,2:end) = (Vdd / 2) * sharp_factor *(1/m) * Delta_1(:,2:end) + ...
                            (lambda/m)*Theta1(:,2:end);
    Theta1n_grad(:,1) = (Vdd / 2) *sharp_factor *(1/m) * Delta_1n(:,1);
    Theta1n_grad(:,2:end) = (Vdd / 2) * sharp_factor *(1/m) * Delta_1n(:,2:end) + ...
                            (lambda/m)*Theta1n(:,2:end);
    Theta2_1_grad(:,1) = sharp_factor *(1/m) * Delta_2_1(:,1);
    Theta2_1_grad(:,2:end) =  sharp_factor * (1/m) * Delta_2_1(:,2:end) + ...
                            (lambda/m)*Theta2_1(:,2:end);
    Theta2_1n_grad(:,1) = sharp_factor *(1/m) * Delta_2_1n(:,1);
    Theta2_1n_grad(:,2:end) = sharp_factor * (1/m) * Delta_2_1n(:,2:end) + ...
                            (lambda/m)*Theta2_1n(:,2:end);
    grad = [Theta1_grad(:) ; Theta1n_grad(:) ; ...
        Theta2_1_grad(:) ; Theta2_1n_grad(:) ; ...
        Theta2_grad(:) ; Theta2n_grad(:)];
elseif (num_hidden_layer == 3)
    
    Theta1_grad(:,1) = (Vdd^2 / 4) * sharp_factor *(1/m) * (Delta_1(:,1));
    Theta1_grad(:,2:end) = (Vdd^2 / 4) * sharp_factor *(1/m) * Delta_1(:,2:end) + ...
                            (lambda/m)*Theta1(:,2:end);
    Theta1n_grad(:,1) = (Vdd^2 / 4) *sharp_factor *(1/m) * Delta_1n(:,1);
    Theta1n_grad(:,2:end) = (Vdd^2 / 4) * sharp_factor *(1/m) * Delta_1n(:,2:end) + ...
                            (lambda/m)*Theta1n(:,2:end);
    Theta2_1_grad(:,1) = (Vdd / 2)* sharp_factor *(1/m) * Delta_2_1(:,1);
    Theta2_1_grad(:,2:end) = ( Vdd / 2) * sharp_factor * (1/m) * Delta_2_1(:,2:end) + ...
                            (lambda/m)*Theta2_1(:,2:end);
    Theta2_1n_grad(:,1) = ( Vdd /2)* sharp_factor *(1/m) * Delta_2_1n(:,1);
    Theta2_1n_grad(:,2:end) = ( Vdd /2) * sharp_factor * (1/m) * Delta_2_1n(:,2:end) + ...
                            (lambda/m)*Theta2_1n(:,2:end);
    Theta2_2_grad(:,1) =  sharp_factor *(1/m) * Delta_2_2(:,1);
    Theta2_2_grad(:,2:end) =  sharp_factor * (1/m) * Delta_2_2(:,2:end) + ...
                            (lambda/m)*Theta2_2(:,2:end);
    Theta2_2n_grad(:,1) =  sharp_factor *(1/m) * Delta_2_2n(:,1);
    Theta2_2n_grad(:,2:end) = sharp_factor * (1/m) * Delta_2_2n(:,2:end) + ...
                            (lambda/m)*Theta2_2n(:,2:end);
    grad = [Theta1_grad(:) ; Theta1n_grad(:) ;...
        Theta2_1_grad(:) ; Theta2_1n_grad(:) ;...
        Theta2_2_grad(:) ; Theta2_2n_grad(:) ;...
        Theta2_grad(:) ; Theta2n_grad(:)];
else
    
    Theta1_grad(:,1) = sharp_factor *(1/m) * (Delta_1(:,1));
    Theta1_grad(:,2:end) = sharp_factor *(1/m) * Delta_1(:,2:end) + ...
                            (lambda/m)*Theta1(:,2:end);
    Theta1n_grad(:,1) = sharp_factor *(1/m) * Delta_1n(:,1);
    Theta1n_grad(:,2:end) = sharp_factor *(1/m) * Delta_1n(:,2:end) + ...
                            (lambda/m)*Theta1n(:,2:end);
    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta1n_grad(:) ; ...
        Theta2_grad(:) ; Theta2n_grad(:)];
end

% -------------------------------------------------------------

% =========================================================================

end
