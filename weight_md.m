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
function g = weight_md(theta,K)
%weight_md returns the gradient of the weight mapping function


g = zeros(size(theta,2),(size(theta,2) * size(theta,1)));

num_n = size(theta,2);
num_pn = size(theta,1); 
g1 = K.*(1.0 ./(1.0 + exp(-theta))).*(1.0 - (1.0 ./(1.0 + exp(-theta))));
g1 = kron(g1,ones(num_n,1))';
theta = 1.0 + (K ./ (1.0 + exp(-theta)));
sum_theta = sum(theta,2);
sums_theta1 = repmat(sum_theta,1,size(theta,2)); % as size of theta
theta = kron(theta,ones(num_n,1))';
sums_theta = kron(sums_theta1,ones(num_n,1))';
sums_theta1 = sums_theta1';
sums_theta1 = 1.0 ./ sums_theta1(:);
diag_num1 = 1:(num_n+1):(num_n^2);
diag_num = [];
for i=1:num_pn
    diag_num = [diag_num (diag_num1+(i-1)*(num_n^2))];
end
g =   ((-theta) ./ ((sums_theta).^2));
g(diag_num) = g(diag_num) + sums_theta1';
g = g .* g1;
% g = weight_mapping(theta,K) - ((weight_mapping(theta,K)).^2);
end
