% ============================================================================%
%                 Main file of ex-situ Training Framework                    %
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
clc;
clear;

 global cc;
 global cc2;
 
cc = [-0.0012   -0.2483   -0.0235   31.7581];   % parameters for activation function from SPICE Vdd = 0.5;
cc2 = [-0.0011    0.2490   -0.0203  193.0602];  % parameters for activation function from SPICE, Vdd = 0.5;
% for other size or loads please use inv_resuts_0.25.mat
% for instance,  for size 3
% load('inv_results_0.25.mat');
% cc = result(95,3:6);
% cc2 = result(95,7:10);
clc;
init_time = tic;

sharp_factor = cc(4);
sharp_factorp = sharp_factor;
sharp_factorn = cc2(4);
% default values
Vdd = 0.5;
sz = 5;
%%% number of layers and number of neurons in each layer
num_hidden_layer = 1;
hidden_layer_size = 10;
%%% if number of hidden layers is 1, the following parameters are not used.
hidden_layer_size2 = 8;
hidden_layer_size3 = 8;
%% INPUT DATA IMPORT: Note: in this way, the TRAINING and TESTING data are the same, it should be corrected later

%%
%%% run face pos data set %%%

load('DataSet/FACE_POS_DATA_full.mat');
num_hidden_layer = 1;
hidden_layer_size = 10;
rand_val = randperm(240, 200);
X = X(:,rand_val);
Y = Y(:,rand_val);
Y = (Y-0.5)./2; % Y is between 0 and 1
X = (X-128)./255/2; % X is between 0 or 255
test_index = randperm(size(X,2),floor(0.2*size(X,2)));
 X_test = ((X(:,test_index))); 
 Y_test = (Y(:,test_index));
 app_name = 'face_pos';
 for i=1:min(size(X,2),16)
    subplot(4,4,i);
    imagesc(reshape(X(:,i),30,32));
    colormap(gray);
    axis off
 end

%%
% %% run FFT %%% x is 0 to 0.5 and y is 0 to 1
% rand_val2 = randperm(11264,7884);
% num_rand = 500;
% rand_val = rand_val2(1:num_rand);
% load('X_fft_A.mat');
% load('Y_fft_A.mat');
% num_hidden_layer = 1;
% hidden_layer_size = 14;
% X = X_fft(:,rand_val) - 0.25;
% Y = (Y_fft(:,rand_val) / 2) / 2;
% % X1 = X_fft ;
% %  y1 = (Y_fft / 2) ;

% %% run FFT %%% x is 0 to 0.5 and y is 0 to 1
% rand_val2 = randperm(11264,7884);
% num_rand = 500;
% rand_val = rand_val2(1:num_rand);
% load('X_fft_A.mat');
% load('Y_fft_A.mat');
% 
% num_hidden_layer = 2;
% hidden_layer_size = 8;
% hidden_layer_size2 = 8;
% % test_index = randperm(11264,floor(0.25*num_rand));
% %  X_test = ((X_fft(:,test_index) - 0.25)); 
% %  Y_test = (Y_fft(:,test_index)./4);
% X = X_fft(:,rand_val) - 0.25;
% Y = (Y_fft(:,rand_val) / 2) / 2;
%% 

% %% run FFT, size=3 %%% x is 0 to 0.5 and y is 0 to 1
% rand_val2 = randperm(11264,7884);
% num_rand = 500;
% sz=3;
% rand_val = rand_val2(1:num_rand);
% load('X_fft_A.mat');
% load('Y_fft_A.mat');
% num_hidden_layer = 1;
% hidden_layer_size = 8;
% X = X_fft(:,rand_val) - 0.25;
% Y = (Y_fft(:,rand_val) / 2) / 2;
% X1 = X_fft ;
%  y1 = (Y_fft / 2) ;
% X1(:,rand_val) = [];
% y1(:,rand_val) = [];
%% 

% %% run blacksholes %%% x is 0 to 1 and y 0 to 0.238
% rand_val2 = randperm(100000,70000);
% num_rand = 700;
% rand_val = rand_val2(1:num_rand);
% load('X_blacksholes_A.mat');
% load('Y_blacksholes_A.mat');
% num_hidden_layer = 1;
% hidden_layer_size = 6;
% test_index = randperm(100000,floor(0.25*num_rand));
%  X_test = ((X_blacksholes(:,test_index) - 0.5) ./ 2); 
%  Y_test = (Y_blacksholes(:,test_index));
%  X = (X_blacksholes(:,rand_val) - 0.5) / 2;
%  Y = (Y_blacksholes(:,rand_val));
%  X1 = X_blacksholes - 0.5 ;
%  y1 = Y_blacksholes  ;
% X1(:,rand_val) = [];
% y1(:,rand_val) = [];

%% run blacksholes size=3%%% x is 0 to 1 and y 0 to 0.238
% rand_val2 = randperm(100000,70000);
% num_rand = 200;
% rand_val = rand_val2(1:num_rand);
% load('X_blacksholes_A.mat');
% load('Y_blacksholes_A.mat');
% num_hidden_layer = 1;
% hidden_layer_size = 8;
% sz=3;
%  X = (X_blacksholes(:,rand_val) - 0.5) / 2;
%  Y = (Y_blacksholes(:,rand_val));
% test_index = randperm(100000,floor(0.25*num_rand));
%  X_test = ((X_blacksholes(:,test_index) - 0.5) ./ 2); 
%  Y_test = (Y_blacksholes(:,test_index));
%  X1 = X_blacksholes - 0.5 ;
%  y1 = Y_blacksholes  ;
% X1(:,rand_val) = [];
% y1(:,rand_val) = [];

%% 

% %% run sobel onelayer%%% -- 0 to 0.7 x and y
% rand_val2 = randperm(100,70);
% num_rand = 70;
% rand_val = rand_val2(1:num_rand);
% load('X_sobel_A.mat');
% load('Y_sobel_A.mat');
% num_hidden_layer = 1;
% hidden_layer_size = 8;
% X = (X_sobel(:,rand_val) - 0.5) / 2;
% Y = ((Y_sobel(:,rand_val))  - 0.5) / 2;
% test_index = randperm(100,floor(0.25*num_rand));
%  X_test = ((X_sobel(:,test_index) - 0.5) ./ 2); 
%  Y_test = (Y_sobel(:,test_index) - 0.5) ./ 2;
%  X1 = X_sobel - 0.5 ;
%  y1 = Y_sobel  - 0.5 ;
% X1(:,rand_val) = [];
% y1(:,rand_val) = [];

%% run sobel 2layers%%% -- 0 to 0.7 x and y
% rand_val2 = randperm(100,70);
% num_rand = 70;
% rand_val = rand_val2(1:num_rand);
% load('X_sobel_A.mat');
% load('Y_sobel_A.mat');
% num_hidden_layer = 2;
% hidden_layer_size = 8;
% hidden_layer_size2 = 8;
% X = (X_sobel(:,rand_val) - 0.5) / 2;
% Y = ((Y_sobel(:,rand_val))  - 0.5) / 2;

% %  X1 = X_sobel - 0.5 ;
%  y1 = Y_sobel  - 0.5 ;
% X1(:,rand_val) = [];
% y1(:,rand_val) = [];
%

%% 
%%% run inversek2j 2layers%%% -- x is -0.5 to 1 and y is -1 to 1
% rand_val2 = randperm(99997,70000);
% num_rand = 700;
% rand_val = rand_val2(1:num_rand);
% load('X_inversek2j_A_norm.mat');
% load('Y_inversek2j_A_norm.mat');
% num_hidden_layer = 1;
% hidden_layer_size = 12; 
% test_index = randperm(99997,floor(0.25*num_rand));
%  X_test = ((X_inversek2j(:,test_index)) ./ 4); 
%  Y_test = (Y_inversek2j(:,test_index)) ./ 4;
% X = X_inversek2j(:,rand_val) ./ 4;
% Y = (Y_inversek2j(:,rand_val)) ./ 4 ;
% X1 = X_inversek2j ./ 2 ;
% y1 = Y_inversek2j  ./ 2 ;
% X1(:,rand_val) = [];
% y1(:,rand_val) = [];

%% 
%% run kmeans %%% -- x is 0.12 to 0.95 and y is 0.1 to 1.08
% rand_val2 = randperm(600,420);
%  num_rand = 420;
%  rand_val = rand_val2(1:num_rand);
% load('X_kmeans_A.mat');
% load('Y_kmeans_A.mat');
% num_hidden_layer = 2;
% hidden_layer_size = 8;
% hidden_layer_size2 = 4;
% test_index = randperm(600,floor(0.25*num_rand));
%  X_test = ((X_kmeans(:,test_index)) - 0.5)./2; 
%  Y_test = ((Y_kmeans(:,test_index)) - 0.6)./2;
%  X = (X_kmeans(:,rand_val) - 0.5)./2;
%  Y = ((Y_kmeans(:,rand_val))  - 0.6)./2;


% %%% run kmeans Identity%%% -- x is 0.12 to 0.95 and y is 0.1 to 1.08
% rand_val2 = randperm(600,420);
%  num_rand = 400;
%  rand_val = rand_val2(1:num_rand);
% load('X_kmeans_A.mat');
% load('Y_kmeans_A.mat');
% % load('kmeans_sig_exact_sigmoid_identity_1_9_2Layers.mat');
% test_index = randperm(600,floor(0.25*num_rand));
%  X_test = (((X_kmeans(:,test_index)) - 0.5))./2; 
%  Y_test = ((Y_kmeans(:,test_index)) - 0.6)./2;
%  X = (X_kmeans(:,rand_val) - 0.5)./2;
%  Y =( (Y_kmeans(:,rand_val))  - 0.6)./2;

%% 
%%% run MNIST %%% -- x is 0 to 1 and y is 0 to 1
% 
% load('MNIST/MNIST_data.mat');
% num_hidden_layer = 1;
% hidden_layer_size = 3;
% rand_val = [1:4,6:9,11:14]; 
%  y =y';
% test_index = randperm(14,floor(0.2*14));
%  X_test = ((X(:,test_index)) - 0.5) ./2; 
%  Y_test = (Y(:,test_index) - 0.5)  ./2;
%  X = (X(:,rand_val) - 0.5 ) /2; 
%  Y = ((y(:,rand_val))  - 0.5) / 2; 
 %%  
 
 %  %%% run CancerData DATA %%% -- x is -1 to 1 and y is 0 to 1
% rand_val =randperm(683,560); 
% load('DataSet/CancerData.mat');
% % load('Cancer_1N.mat');
% num_hidden_layer = 1;
% hidden_layer_size = 1;
% Y = double(Y);
%  Y(find(Y == 0)) = -1;
%  Y =Y';
%  X = X_norm';
% test_index = randperm(683,floor(0.2*683));
% %  X_test = ((X(:,test_index)) - 0.5) ./2; 
% %  Y_test = (Y(:,test_index))  ./4;
% 
%  X = ((X(:,rand_val)) - 0.5) ./2; 
%  Y = (Y(:,rand_val))  ./4;
%  X_test = X;
%  Y_test = Y;

 %%% run Iris DATA %%% -- x is -1 to 1 and y is 0 to 1
% rand_val =randperm(150,130); 
%  
% % rand_val = 1:70;
% num_hidden_layer = 1;
% hidden_layer_size = 3;
% load('IrisData.mat');
% Y = double(Y);
%  Y(find(Y == 0)) = -1;
%  Y =Y';
%  X = X_norm';
% test_index = randperm(150,20);
%  X_test = ((X(:,test_index)) - 0.5) ./2; 
%  Y_test = (Y(:,test_index))  ./4;
%  X = ((X(:,rand_val)) - 0.5) ./2; 
%  Y = (Y(:,rand_val))  ./4;


% run IoT MHEALTH_DATA %%% -- x is -1 to 1 and y is 0 to 1
% rand_val =randperm(161279,1000); 
% % load('IoT/MHEALTHDATA_20_8.mat');
% load('IoT/MHEALTHDAT_1.mat');
% 
% Y = double(Y);
%  Y(find(Y == 0)) = -1;
%  Y=Y';
%  X = X_norm';
%   rand_val =randperm(161279,800); 
%  X = (X(:,rand_val)); 
%  Y = (Y(:,rand_val))  ./4;
% num_hidden_layer = 1;
% hidden_layer_size = 50;
% X_test = X(:,1:20);
% Y_test = Y(:,1:20);
% 
% hidden_layer_size2 = 8;

%% AND2 GATE without hidden layer
% X = [1 1; 0 0;1 0; 0  1]';
% Y = [1, 0,0,0];
% X = (X-0.5)./2;
% Y = (Y-0.5)./2;
% X_test = X;
% Y_test = Y;
%  num_hidden_layer = 0;

% =================================================================

% =============================== Train the circuit==================================

lambda = 0; % regularization term
J = zeros(4,3);
max_iter = 20000; % maximum iteration to run
iter_loop = 1000; % number of iterations per loop
%%% number of neurons in first and last layer
input_layer_size = size(X,1);
num_labels = size(Y,1);

stop_sign1 = zeros((max_iter/iter_loop),1);
test_error = zeros((max_iter/iter_loop),1);
%%%for weights constraind %%%
         % K(1) for Theta1_mapping , K(2) for Theta_2_1 mapping if existed
         % , K(3) for Theta_2_2 mapping if existed and K(4) for Theta_2 mapping
         K = 7.33 * [1, 1, 1, 1];
        %%%finish %%%

         fprintf('run BackPropagation for %d neurons.\n', hidden_layer_size);
            [Theta1, Theta1n, Theta2, Theta2n, Theta2_1, Theta2_1n,...
                            Theta2_2, Theta2_2n, J(1,1),...
                            stop_sign1(1)] = ...
                            my_backpropagation(Vdd, K, num_hidden_layer,input_layer_size,...
                                hidden_layer_size, hidden_layer_size2,...
                                hidden_layer_size3, num_labels, ...
                                X, Y,...
                                sharp_factor,sharp_factorn, iter_loop, lambda);
%         [Theta1, Theta1n, Theta2, Theta2n, Theta2_1, Theta2_1n,...
%                             Theta2_2, Theta2_2n, J(1,1),...
%                             stop_sign1(1)] = ...
%                             my_backpropagation(Vdd, K, num_hidden_layer,input_layer_size,...
%                                 hidden_layer_size, hidden_layer_size2,...
%                                 hidden_layer_size3, num_labels, ...
%                                 X, Y,...
%                                 sharp_factorp,sharp_factorn, iter_loop, lambda);
                            iter=1;
%                             fprintf('\nAccuracy after %d itration is :%d \n',((iter)*iter_loop),stop_sign1(iter));
 for iter= 2:(max_iter/iter_loop)
     
%      test_error(iter-1) = nn_testing(  num_hidden_layer, ...
%                                    input_layer_size, ...
%                                    hidden_layer_size, ...
%                                    num_labels, ...
%                                    X1, y1, lambda,sharp_factor,...
%                                    B_Theta1, B_Theta2, B_Theta2_1, Theta2_2);
%          fprintf('\nTesting Set Accuracy: %f\n', test_error(iter-1)); 
        if(stop_sign1(iter-1) == 100)
            break
        end
                 Theta1_init = Theta1 ;
         Theta1n_init = Theta1n ;
         Theta2_init = Theta2 ;
         Theta2n_init = Theta2n ;
         Theta2_1_init = Theta2_1 ;
         Theta2_1n_init = Theta2_1n ;
        % Theta2_2_init = Theta2_2 ;
        %fprintf('run BackPropagation for %d neurons.\n', hidden_layer_size);
        try
        [Theta1, Theta1n, Theta2, Theta2n, Theta2_1, Theta2_1n,...
                            Theta2_2, Theta2_2n, J(1,1),...
                            stop_sign1(iter)] = ...
                            my_backpropagation(Vdd,K, num_hidden_layer,input_layer_size,...
                                hidden_layer_size, hidden_layer_size2,...
                                hidden_layer_size3, num_labels, ...
                                X, Y,...
                                sharp_factor, sharp_factorn, iter_loop, lambda,...
                                0,Theta1_init, Theta1n_init, ...
                                Theta2_init, Theta2n_init,...
                                Theta2_1_init,  Theta2_1n_init);
        catch
            break;
        end
%          Theta1_init = Theta1 ;
%          Theta1n_init = Theta1n ;
%          Theta2_init = Theta2 ;
%          Theta2n_init = Theta2n ;
%          Theta2_1_init = Theta2_1 ;
%          Theta2_1n_init = Theta2_1n ;
%         % Theta2_2_init = Theta2_2 ;
%         %fprintf('run BackPropagation for %d neurons.\n', hidden_layer_size);
%         [Theta1, Theta1n, Theta2, Theta2n, Theta2_1, Theta2_1n,...
%                             Theta2_2, Theta2_2n, J(1,1),...
%                             stop_sign1(iter)] = ...
%                             my_backpropagation(Vdd, K, num_hidden_layer,input_layer_size,...
%                                 hidden_layer_size, hidden_layer_size2,...
%                                 hidden_layer_size3, num_labels, ...
%                                 X, Y,...
%                                 sharp_factorp, sharp_factorn, iter_loop, lambda,...
%                                 0,Theta1_init, Theta1n_init, ...
%                                 Theta2_init, Theta2n_init,...
%                                 Theta2_1_init,  Theta2_1n_init);
%                             
    fprintf('\nAccuracy after %d itration is :%d \n',((iter)*iter_loop),stop_sign1(iter));
 end
    fprintf('Training finished\nElapsed time to now: %fs',toc(init_time));
    
    %% ================ Weights Mapping and MSE in matlab ==============
%%% if you will run the HSPICE part, you must skip this part    

%%% Note: This section is necessary after running training to convert weights
%%% to resistance

%% ----------------------------------------------
    Theta2_1 = weight_mappingsig(Theta2_1,K(2));
    Theta2_2 = weight_mappingsig(Theta2_2,K(3));
    Theta2 = weight_mappingsig(Theta2,K(4));
    Theta1 = weight_mappingsig(Theta1,K(1));
    Theta2_1n = weight_mappingsig(Theta2_1n,K(2));
    Theta2_2n = weight_mappingsig(Theta2_2n,K(3));
    Theta2n = weight_mappingsig(Theta2n,K(4));
    Theta1n = weight_mappingsig(Theta1n,K(1));

    
%% ----------------------------------------------

pred = predictA(Theta1, Theta1n, Theta2, Theta2n, [],  [],...
                [], [],...
                X', num_labels, sharp_factor, sharp_factorn,...
                num_hidden_layer,K, Vdd);
[~,preds] = max(pred,[],2);
[~,ynumber] = max(Y',[],2);
m = size(X,2);
J = (1/m)*sum(sum( (pred' - Y).^2 ) ) 
num = sum(sum(preds ~= ynumber))
Error_CA = num/m
fprintf('Software MSE = %f | Accuracy = %d\n' ,  J, 100*(1-Error_CA));
%=============================== end training =========================
%% ======================== Test and Evaluate the circuit ================

Theta1_all = zeros(size(Theta1,1), size(Theta1,2)*2);
Theta1_all(:, 1:2:end) = Theta1;
Theta1_all(:,2:2:end) = Theta1n;
% 
% % % one layer

if(exist('Theta2_1','var') && exist('Theta2_1n','var') && ~isempty(Theta2_1))
    
    Theta1_all = zeros(size(Theta1,1), size(Theta1,2)*2);
    Theta1_all(:, 1:2:end) = Theta1;
    Theta1_all(:,2:2:end) = Theta1n;
    
    Theta2_1_all = zeros(size(Theta2_1,1), size(Theta2_1,2)*2);
    Theta2_1_all(:, 1:2:end) = Theta2_1;
    Theta2_1_all(:,2:2:end) = Theta2_1n;
    
    Theta2_all = zeros(size(Theta2,1), size(Theta2,2)*2);
    Theta2_all(:, 1:2:end) = Theta2;
    Theta2_all(:,2:2:end) = Theta2n;
    
    %======== necessary after running training ======
    Theta1_all = weight_mappingsig(Theta1_all, K(1));
    Theta2_all = weight_mappingsig(Theta2_all, K(4));
    Theta2_1_all = weight_mappingsig(Theta2_1_all, K(2));
%     Theta2_2_all = weight_mappingsig(Theta2_2_all, K(3));
    %==============================================
    
    
    
    weights = {Theta1_all(:,3:end); Theta2_1_all(:,3:end); Theta2_all(:,3:end)};
    b = {Theta1_all(:,1:2); Theta2_1_all(:,1:2); Theta2_all(:,1:2)};
    
else
    Theta1_all = zeros(size(Theta1,1), size(Theta1,2)*2);
    Theta1_all(:, 1:2:end) = Theta1;
    Theta1_all(:,2:2:end) = Theta1n;
    
    Theta2_all = zeros(size(Theta2,1), size(Theta2,2)*2);
    Theta2_all(:, 1:2:end) = Theta2;
    Theta2_all(:,2:2:end) = Theta2n;
     % necessary after running training (for spice simulation)
    Theta1_all = weight_mappingsig(Theta1_all, K(1));
    Theta2_all = weight_mappingsig(Theta2_all, K(4));
    %==============================================
    Theta2_1=[];
    Theta2_1n=[];
    weights = {Theta1_all(:,3:end); Theta2_all(:,3:end)};
    b = {Theta1_all(:,1:2); Theta2_all(:,1:2)};
end

%======================== START Testing ============================
filename = 'test1';
X = X_test;
Y = Y_test;
clear out_spice
nn2spice_analog([filename '.sp'],X , 1, weights, b,Vdd,sz)
debug_s = readmt([filename '.mt0']);
% 
out_matlab = predictA(Theta1, Theta1n,  Theta2, Theta2n, ...
    Theta2_1, Theta2_1n, [],[], X',  size(Y,1), sharp_factorp, sharp_factorn, ...
    length(weights)-1, K, Vdd);
% 
for k=1:size(Y,1)
    out_spice(:,k) = sig(debug_s, sprintf('outn%d',k));
end
m = numel(Y);
app_mse_matlab = sum(sum((out_matlab-Y').^2))./m
 app_mse = sum(sum((out_spice-Y').^2))./m
%% ================  Extra Simulation for Delay, Power, and Energy Measurment ===================
good_indx_rel_err = find(abs((out_spice(:,1)'-Y(1,:))./Y(1,:))<0.1);

t_hold = 200e-9; %the time holding each input (to propagate)
% X_sel = X(:,good_indx_rel_err);
% Y_sel = Y(1,good_indx_rel_err);
try
X_sel = X(:,1:10);
Y_sel = Y(:,1:10);
catch
X_sel = X;
Y_sel = Y; 
end
if(length(good_indx_rel_err)>10)
    nn2spice_analog_tran([filename '.sp'],X_sel(:,1:10) , 1, weights, b, Y_sel(:,1:10),Vdd, t_hold);
    LL = 10;
else
    nn2spice_analog_tran([filename '.sp'],X_sel , 1, weights, b, Y_sel,Vdd, t_hold);
    LL = length(X_sel);
end


s_tr = loadsig([filename '_tran.tr0']); %note that _tran is added to file name in nn2spice_analog_tran
t = sig(s_tr, 'TIME');
while((sum(diff(t)==0))~=0)
    repeat_indx=find((diff(t)==0));
    t(repeat_indx+1)=t(repeat_indx)+1e-15;
end
input_index = 1;
% input_index = 40; % for MNIST
sig_in = sig(s_tr, sprintf('inp1_%d',input_index));
clear sig_out;
for k=1:size(Y,1)
    sig_out(k,:)  = sig(s_tr, sprintf('outndummy%d',k));
end
t1 = t((abs(diff(sig_in))>0.001));
t2_in=t1(diff(t1)>1e-9);
% measure delay
for k=1:size(Y,1)
    for i=1:length(t2_in)-1 %for each input edge
        x1 = t2_in(i);
        y1 = sig_out(k,t==x1);
        t_settle = t2_in(i+1)-20e-12; % 10ps before edge
        y2 =  spline(t,sig_out(k,:),t_settle);
        y_end = (y2-y1)*0.9+y1;
        y_end2 = (y2-y1)*1.1+y1; %for overshoot
        % find where y reaches the 0.9 of its final value
        td_indx = find((t((t<t_settle))>x1));
        t_good = t(td_indx);
        y_diff = abs(sig_out(k,td_indx)-y_end);
        y_diff2 = abs(sig_out(k,td_indx)-y_end2);
        DataInv = 1.01*max(y_diff) - y_diff;
        %find all points where y=0.9y_end
        if(~isempty(DataInv))
            [Minima,MinIdx] = findpeaks(DataInv);
        else
            MinIdx = [];
        end
        % find exactly the last cross in the range
        
        DataInv2 = 1.01*max(y_diff2) - y_diff2;
        if(~isempty(DataInv2))
            %find all points where y=1.1y_end
            [Minima2,MinIdx2] = findpeaks(DataInv2);
        else
            MinIdx2 =[];
        end
        
        x2 = t_good(max(max([MinIdx, MinIdx2])));
        if(~isempty(x2) && ~isempty(x1))
        d{k}(i) = x2-x1;
        end
        
    end
        if(~isempty(d{k}))
        d_good{k} = d{k}(d{k}<2e-7);
    end
    
    if(~isempty(d_good{k}))
        delay_max(k) = max(d_good{k});
        delay_mean(k) = mean(d_good{k});
    end
end


    semi_t(1:2:size(Y_sel,2)*2) = (0:size(Y_sel,2)-1)*t_hold;
    semi_t(2:2:size(Y_sel,2)*2) = (1:size(Y_sel,2))*t_hold;
    semi_Y = zeros(size(Y_sel,1),size(Y_sel,2)*2);
    semi_Y(:,1:2:size(Y_sel,2)*2) = Y_sel;
    semi_Y(:,2:2:size(Y_sel,2)*2) = Y_sel;

    figure;
    plot(t,sig_out(1,:))
    hold on;
    plot(semi_t, semi_Y,'color','red');
    title(sprintf('MSE=%e',app_mse));
    legend('Circuit Output', 'Target');

t_hold = max(delay_max)*1.05;

[delay, del_indx] = max(delay_max);
if(length(good_indx_rel_err)>10)
    nn2spice_analog_tran([filename '.sp'],X_sel(:,1:10) , 1, weights, b, Y_sel(:,1:10),Vdd, delay);
else
    nn2spice_analog_tran([filename '.sp'],X_sel , 1, weights, b, Y_sel,Vdd, delay);
end
tran_s = readmt([filename '_tran.mt0']);
p_tot = sig(tran_s, 'ptot');
energy = abs(sig(tran_s, 'evdd'));
disp(sprintf('mse\t\t\t\tpower\t\t\tdelay\t\t\tenergy'));
disp(sprintf('%e\t%e\t%e\t%e',app_mse, p_tot, delay, energy));
nnConfig = sprintf('%d',size(X,1));
for k=1:length(weights)
    nnConfig = strcat(nnConfig, sprintf('-%d',size(weights{k},1)));
end
disp(['nnConfig:', nnConfig]);
s_tr2 = loadsig([filename '_tran.tr0']);
nnResult = [app_mse, p_tot, delay, energy];
figure;
plot(sig(s_tr2,'TIME'),sig(s_tr2, sprintf('outndummy%d',del_indx)))

% ===============================================================================================
%% 
[~,preds] = max(out_spice,[],2);
[~,ynumber] = max(Y',[],2);
m = size(X,2);
num = sum(sum(preds ~= ynumber))
Error_rate = num/m;
fprintf('MSE = %f | Accuracy = %d\n' ,  app_mse, 100*(1-Error_rate));

fprintf('Spice simulation finished\nTotal elapsed time: %fs\n',toc(init_time));
