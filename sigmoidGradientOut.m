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
function g = sigmoidGradientOut(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, it should return
%   the gradient for each element.

g = zeros(size(z));


% g = sigmoid(z).*(1-sigmoid(z));

g = (1-(tansig(z).^2));









% =============================================================




end
