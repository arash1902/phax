 function [v, delay, power, energy] = nn2spice_analog(filename, data_in, num_proc,  varargin)

% usage1: nn2spice_inv_diff(filename, data_in, num_proc,  layers, weights, biases)
% usage2: nn2spice_inv_diff(filename, data_in, num_proc,  nn_Object)
% nn_Object: the neural network object created by nntool or nftool
% layers: 1xN array containing the number of neurons of each layer where N-1
% is the number of layers
% weights: 1xN-1 cell array, cell i containg the weights from layer(i) to
% layer(i+1)
% biases: 1xN-1 array containing the bias of each layer
% num_proc is the number of processors is spice -mp
% vdd_logic = 1;
identity=0; 
if(size(varargin,2) == 4)
         weights = varargin{1};
     biases  = varargin{2};
     vdd_logic  = varargin{3};
     sz  = varargin{4};
    for jk = 1:length(weights)
        layers(jk) = size(weights{jk},2)/2;
    end
    layers(length(weights)+1) = size(weights{jk},1);
elseif(size(varargin,2) == 5)
     layers  = varargin{1};
     weights = varargin{2};
     biases  = varargin{3};
     vdd_logic  = varargin{4};
else
    nn = varargin{1};
    in_size  = nn.inputs{1}.size;
    layers_size = [];
    for i=1:length(nn.layers)
        layers_size = [layers_size, nn.layers{i}.size];
    end
    layers = [in_size, layers_size];
    weights = {nn.IW{1}};
    for j = 2:length(nn.IW)
        weights = {weights{:},  nn.LW{j,j-1}};
    end
    biases = nn.b;
    if(size(varargin,2) == 2)
        targets = varargin{2};
    end
    if(size(varargin,2) == 3)
        targets = varargin{2};
        vdd_logic =  varargin{3};
    end
end
% weight2r_analog(weights, biases, filename);
weight2x_analog(weights, biases, filename);
fid = fopen(filename, 'w');
period = 100; % input data period in ns
fprintf(fid, '\n .param vdd_train=2.5');
fprintf(fid, '\n .param parvdd_logic=%e',vdd_logic);
fprintf(fid, '\n .hdl ''memr_generalized.va''\n');
fprintf(fid, '*========== DVI parameters =====================');
fprintf(fid, '\n.param	PER=%d	*%d ns PERIOD\n',period, period);
fprintf(fid, '\n+		Vplus=''parvdd_logic/2''');
fprintf(fid, '\n+		Vminus=''-parvdd_logic/2''');
fprintf(fid, '\n+		SLP=''PER/20''');
% fprintf(fid, '\n+		VH = ''(VDD/2)*0.5''');
% fprintf(fid, '\n+		VL = ''(-VDD/2)*0.5''');
fprintf(fid, '\n+		VARDELAY = ''PER*0.9''');
fprintf(fid, '\n+		ACSS_DEL = ''SLP''');
fprintf(fid, '\n*===============================================');
%============== instantiate layer subcircuits =====================
for layer_num=1:(length(layers)-1)  
   fprintf(fid, '\n xLayer%d   ',layer_num);
   
   % print the input nodes 
   fprintf(fid, 'b0 b1 '); %the bias is shared for all layers
   for i=1:layers(layer_num)
       fprintf(fid, 'inp%d_%d   ',layer_num, i);
       fprintf(fid, 'inn%d_%d   ',layer_num, i); 
   end
   
   % print the output nodes 
   if(layer_num ~= length(layers)-1)
       for i=1:layers(layer_num+1)
           fprintf(fid, 'inp%d_%d   ',layer_num+1, i);
           fprintf(fid, 'inn%d_%d   ',layer_num+1, i);
       end
   else
      for i=1:layers(layer_num+1)
           fprintf(fid, 'outpdummy%d   ', i);
           fprintf(fid, 'outndummy%d   ', i);
       end
   end 
   fprintf(fid, 'vdd_logic vss layer%d   ',layer_num);
end
  fprintf(fid, '\n.inc ''comp.sp''\n');
  for i=1:layers(layer_num+1)
       fprintf(fid, 'xcp%d outndummy%d  outp%d vdd_logic vss inv_diff  \n', i, i, i);
       fprintf(fid, 'xcn%d outpdummy%d  outn%d vdd_logic vss inv_diff \n', i, i, i);
       
       %the resistive load for output
%        fprintf(fid, 'routp%d outndummy%d  0  1e6\n', i, i);
%        fprintf(fid, 'routn%d outpdummy%d  0  1e6\n', i, i);
   end
fclose(fid);

%===================== define layer subcircuits ==========================
for layer_num=1:(length(layers)-1)
    Nin = layers(layer_num);
    Nout = layers(layer_num+1);
    if(layer_num==length(layers)-1 && identity== 1)  % for identity outputs
        make_layer_flexible(layer_num, Nin, Nout, filename, sz, 4); 
    else
        make_layer_flexible(layer_num, Nin, Nout, filename, sz, 1); 
    end
    
end


% ===================== create inputs txt  ============================
fid2 = fopen(sprintf('inputs_%s.txt', filename),'w');
[~, Nin] = size(weights{1});
Nin = Nin/2;
% the header row
% for jj=1:Nin
%    fprintf(fid2, ' x%d',jj);
% end

[~, Nsamples] = size(data_in);
for i=1:Nsamples
    for jj=1:Nin
        fprintf(fid2, ' %3.2e',data_in(jj,i));
    end
    fprintf(fid2, '\n');
end
fclose(fid2);
%====================================================================

fid = fopen(filename, 'a');
%========================= simulation commands ============================
 fprintf(fid, '\nvvdd_logic vdd_logic 0 	''parvdd_logic/2''\n');
 fprintf(fid, '\nvvss 0 vss 	''parvdd_logic/2''\n');
%  fprintf(fid, '\n.GLOBAL vdd_logic\n');
fprintf(fid, '\n .DATA inputData MER\n FILE=''inputs_%s.txt'' ', filename);

[~, Nin] = size(weights{1});
Nin = Nin./2;
for ki = 1:Nin
       fprintf(fid, ' xv%d=%d', ki, ki);
end
fprintf(fid, '\n.ENDDATA\n');


for ki = 1:Nin
%        fprintf(fid, '\n .param xv%d=0', ki);
       fprintf(fid, '\n .param xvp%d=''xv%d''', ki, ki);
       fprintf(fid, '\n .param xvn%d=''-xv%d''', ki, ki);
end
% =================== input sources ==================================
fprintf(fid, '\n vb0 b0 0 ''-parvdd_logic/2'' \n');
fprintf(fid, '\n vb1 b1 0 ''parvdd_logic/2'' \n'); %vdd_logic = 1V
for iii=1:Nin
    fprintf(fid, '\nvxp%d inp1_%d 0 ''xvp%d''', iii, iii, iii);
    fprintf(fid, '\nvxn%d inn1_%d 0 ''xvn%d''', iii, iii, iii);
end
[Nout, ~] = size(weights{end});
% DEBUGGING ========================
for iii = 1:4
    fprintf(fid, '\n.MEAS TRAN net1_%d AVG v(xLayer1.in%d) from=99.9999n to=100n', iii,  iii);
end
for iii = 1:4
    fprintf(fid, '\n.MEAS TRAN l1op_%d AVG v(xLayer1.outp%d) from=99.9999n to=100n', iii,  iii);
    fprintf(fid, '\n.MEAS TRAN L1on_%d AVG v(xLayer1.outn%d) from=99.9999n to=100n', iii,  iii);
end
for iii = 1:4
    fprintf(fid, '\n.MEAS TRAN net2_%d AVG v(xLayer2.in%d) from=99.9999n to=100n', iii,  iii);

end
for iii = 1:4
        fprintf(fid, '\n.MEAS TRAN L2op_%d AVG v(xLayer2.outp%d) from=99.9999n to=100n', iii,  iii);
    fprintf(fid, '\n.MEAS TRAN L2on_%d AVG v(xLayer2.outn%d) from=99.9999n to=100n', iii,  iii);
    fprintf(fid, '\n.MEAS TRAN igate_%d AVG v(xLayer2.outn%d) from=99.9999n to=100n', iii,  iii);
end
% ENDDEBUGGING ========================

for iii=1:Nout
    fprintf(fid, '\n.MEAS TRAN outp%d AVG v(outpdummy%d) from=99.9999n to=100n', iii, iii);
    fprintf(fid, '\n.MEAS TRAN outn%d AVG v(outndummy%d) from=99.9999n to=100n', iii, iii);
    fprintf(fid, '\n.MEAS TRAN sigmaO%d AVG v(xLayer%d.in%d) from=99.9999n to=100n', iii, length(weights), iii);
    
    
    fprintf(fid, '\n.MEAS TRAN xp%d AVG v(inp1_%d) from= 99.9999n  to=100n', iii, iii);
    fprintf(fid, '\n.MEAS TRAN xn%d AVG v(inn1_%d) from=99.9999n to=100n', iii, iii);
    fprintf(fid, '\n.MEAS TRAN xp2%d AVG v(inp2_%d) from=99.9999n to=100n', iii, iii);
    fprintf(fid, '\n.MEAS TRAN xn2%d AVG v(inn2_%d) from=99.9999n to=100n', iii, iii);
end
% fprintf(fid, '\n.MEAS TRAN net11 AVG v(xLayer1.in1) from=1n to=0.99999n');
% fprintf(fid, '\n.MEAS TRAN net12 AVG v(xLayer1.in2) from=1n to=0.99999n');
% fprintf(fid, '\n.MEAS TRAN net13 AVG v(xLayer1.in3) from=1n to=0.99999n');
% fprintf(fid, '\n.VEC ''inputs_%s.txt''', filename);
%fprintf(fid, '\n.tran 100p %dn sweep parvdd_logic lin 1 0.5 %e\n', period*Nsamples, vdd_logic);
fprintf(fid, '\n.tran 10p %dn sweep data=inputData \n', period);
% fprintf(fid, '\n.tran 10p %dn\n', period);
% fprintf(fid, '\n.tran 100p %dn\n', period*Nsamples);

% ============================= MEASURES =============================
fprintf(fid, '\n.MEAS TRAN p_tot AVG SRC_PWR \n');
fprintf(fid, '\n.meas tran DLH trig V(inp1_1) val=''0'' rise=1 targ v(outp1) val=''0'' rise=1');
fprintf(fid, '\n.meas tran DHL trig V(inp1_1) val=''0'' rise=2 targ v(outp1) val=''0'' fall=1');
fprintf(fid, '\n.meas delay PARAM = ''(DHL+DLH)/2''');
fprintf(fid, '\n.meas energy PARAM = ''delay*p_tot''');
  
%  fprintf(fid,'\n.Variation \n.Local_Variation');
%  fprintf(fid,'\n x memristor Rinit=10%%');
%  fprintf(fid,'\n.End_Local_Variation');
%  fprintf(fid,'\n.End_Variation');
 fprintf(fid,'\n.OPTIONS ACCURATE POST=2');
% fprintf(fid,'\n.OPTION METHOD=GEAR'n);
fprintf(fid,'\n.END');
fclose(fid);
fname = strtok(filename,'.');
command = sprintf('hspice -i %s -o %s.lis -mp %d', filename, fname, num_proc);
tic
system(command);
toc
fname = strtok(filename,'.');
% s_mt = readmt([fname '.mt0']);
% 
% delay = sig(s_mt, 'delay');
% power = sig(s_mt, 'p_tot');
% energy = sig(s_mt, 'energy');
% v = sig(s_mt, 'parvdd_logic');

% s_tr = loadsig([fname '.tr0']);
% t = sig(s_tr, 'TIME');
% hold off;
% plot(t,sig(s_tr, 'inp1_1'), 'blue');
% hold on;
% plot(t,sig(s_tr, 'outp1'), 'red');
% legend('Vin[0]', 'Vout');
end