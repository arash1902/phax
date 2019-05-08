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
function g = weight_mapping(theta,K)
%%%for weights constraind %%%
%weight mapping Compute weight mapping function
%   J = SIGMOID(z) computes the sigmoid of z.
theta = 1.0 + (K ./ (1.0 + exp(-theta)));
sum_theta = sum(theta,2);
g = theta ./ (repmat(sum_theta,1,size(theta,2)));
end
