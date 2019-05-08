 function [v, delay, power, energy] = nn2spice_analog_tran(filename, data_in, num_proc,  varargin)

% usage1: nn2spice_inv_diff(filename, data_in, num_proc,  layers, weights, biases)
% usage2: nn2spice_inv_diff(filename, data_in, num_proc,  nn_Object)
% nn_Object: the neural network object created by nntool or nftool
% layers: 1xN array containing the number of neurons of each layer where N-1
% is the number of layers
% weights: 1xN-1 cell array, cell i containg the weights from layer(i) to
% layer(i+1)
% biases: 1xN-1 array containing the bias of each layer
% num_proc is the number of processors is spice -mp
sz=5;
identity=0;
fname = strtok(filename,'.');
filename = strcat(fname, '_tran.sp');
fname = strtok(filename,'.');
if(size(varargin,2) == 5)
         weights = varargin{1};
     biases  = varargin{2};
     
    for jk = 1:length(weights)
        layers(jk) = size(weights{jk},2)/2;
    end
    layers(length(weights)+1) = size(weights{jk},1);
    targets = varargin{3};
    vdd_logic = varargin{4};
    thold = varargin{5};
elseif(size(varargin,2) == 6)
     layers  = varargin{1};
     weights = varargin{2};
     biases  = varargin{3};

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

fprintf(fid, '\n .param vdd_train=2.5');
fprintf(fid, '\n .param parvdd_logic=%e',vdd_logic);
fprintf(fid, '\n .hdl ''memr_generalized.va''\n');

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
% thold = 40e-9;
trise = 20e-12;
period = thold+trise; % input data period
[~, Nsamples] = size(data_in);
for i=1:Nin
    t_now = 0;
    fprintf(fid2, 'vxp%d inp1_%d 0 PWL ', i, i);
    for jj=1:Nsamples
        fprintf(fid2, '%e %e ',t_now, data_in(i,jj));
        t_now = t_now + thold;
        fprintf(fid2, '%e %e ',t_now, data_in(i,jj));
        t_now = t_now + trise;
    end
    fprintf(fid2, '\n');
    
    t_now = 0;
    fprintf(fid2, 'vxn%d inn1_%d 0 PWL ', i, i);
    for jj=1:Nsamples
        fprintf(fid2, '%e %e ',t_now, -data_in(i,jj));
        t_now = t_now + thold;
        fprintf(fid2, '%e %e ',t_now, -data_in(i,jj));
        t_now = t_now + trise;
    end
    fprintf(fid2, '\n');
end
fclose(fid2);
%====================================================================

fid = fopen(filename, 'a');
%========================= simulation commands ============================
 fprintf(fid, '\nvvdd_logic vdd_logic 0 	''parvdd_logic/2''\n');
 fprintf(fid, '\nvvss  vss 0	''-parvdd_logic/2''\n');
 
% =================== input sources ==================================
fprintf(fid, '\n vb0 b0 0 ''-parvdd_logic/2'' \n');
fprintf(fid, '\n vb1 b1 0 ''parvdd_logic/2'' \n'); %vdd_logic = 1V
fprintf(fid, '\n .include ''inputs_%s.txt'' \n',filename);
% for iii=1:Nin
%     fprintf(fid, '\nvxp%d inp1_%d 0 ''xvp%d''', iii, iii, iii);
%     fprintf(fid, '\nvxn%d inn1_%d 0 ''xvn%d''', iii, iii, iii);
% end
[Nout, ~] = size(weights{end});
% ================== DEBUGGING ========================
% for iii = 1:4
%     fprintf(fid, '\n.MEAS TRAN net1_%d AVG v(xLayer1.in%d) from=9.99999n to=10n', iii,  iii);
% end
% for iii = 1:4
%     fprintf(fid, '\n.MEAS TRAN l1op_%d AVG v(xLayer1.outp%d) from=9.99999n to=10n', iii,  iii);
%     fprintf(fid, '\n.MEAS TRAN L1on_%d AVG v(xLayer1.outn%d) from=9.99999n to=10n', iii,  iii);
% end
% for iii = 1:4
%     fprintf(fid, '\n.MEAS TRAN net2_%d AVG v(xLayer2.in%d) from=9.99999n to=10n', iii,  iii);
% 
% end
% for iii = 1:4
%         fprintf(fid, '\n.MEAS TRAN L2op_%d AVG v(xLayer2.outp%d) from=9.99999n to=10n', iii,  iii);
%     fprintf(fid, '\n.MEAS TRAN L2on_%d AVG v(xLayer2.outn%d) from=9.99999n to=10n', iii,  iii);
%     fprintf(fid, '\n.MEAS TRAN igate_%d AVG v(xLayer2.outn%d) from=9.99999n to=10n', iii,  iii);
% end
% ================== ENDDEBUGGING ========================

% for iii=1:Nout
%     fprintf(fid, '\n.MEAS TRAN outp%d AVG v(outpdummy%d) from=9.99999n to=10n', iii, iii);
%     fprintf(fid, '\n.MEAS TRAN outn%d AVG v(outndummy%d) from=9.99999n to=10n', iii, iii);
%     fprintf(fid, '\n.MEAS TRAN sigmaO%d AVG v(xLayer%d.in%d) from=9.99999n to=10n', iii, length(weights), iii);
%     
%     
%     fprintf(fid, '\n.MEAS TRAN xp%d AVG v(inp1_%d) from= 9.99999n  to=10n', iii, iii);
%     fprintf(fid, '\n.MEAS TRAN xn%d AVG v(inn1_%d) from=9.99999n to=10n', iii, iii);
%     fprintf(fid, '\n.MEAS TRAN xp2%d AVG v(inp2_%d) from=9.99999n to=10n', iii, iii);
%     fprintf(fid, '\n.MEAS TRAN xn2%d AVG v(inn2_%d) from=9.99999n to=10n', iii, iii);
% end

% % fprintf(fid, '\n.MEAS TRAN net11 AVG v(xLayer1.in1) from=1n to=0.99999n');
% % fprintf(fid, '\n.MEAS TRAN net12 AVG v(xLayer1.in2) from=1n to=0.99999n');
% % fprintf(fid, '\n.MEAS TRAN net13 AVG v(xLayer1.in3) from=1n to=0.99999n');
% % fprintf(fid, '\n.VEC ''inputs_%s.txt''', filename);
% %fprintf(fid, '\n.tran 100p %dn sweep parvdd_logic lin 1 0.5 %e\n', period*Nsamples, vdd_logic);
% % fprintf(fid, '\n.tran 10p %dn sweep data=inputData \n', period);
fprintf(fid, '\n.MEAS TRAN ptot AVG SRC_PWR \n');
fprintf(fid, '\n.MEAS TRAN evdd  INTEG PAR(''((V(vdd_logic)*i(vvdd_logic)+V(vss)*i(vvss)))/%d'')\n', Nsamples);
fprintf(fid, '\n.tran 10p %d\n', period*Nsamples);


% fprintf(fid, '\n.tran 100p %dn\n', period*Nsamples);

% ============================= MEASURES =============================

% fprintf(fid, '\n.MEAS TRAN p_vdd AVG PAR(''p(vvdd_logic)+P(vvss)'' \n');
% tfrom = 2e-7;
% tto = 2.2e-7;

delay_samples = 3;

for jj=1:Nout
    for i=2:delay_samples+1
        vin_cross = (data_in(1,i)+data_in(1,i-1))./2;
        vout_cross = (targets(jj,i)+targets(jj,i-1))./2;


%         fprintf(fid, '\n.probe tran der%d=deriv(''v(outndummy%d)'')', jj, jj);    
        cross_out = 1;
        cross_in = 1;
        % check for previous crosses
        for j=2:i-1
           if((vout_cross>targets(jj,i) && vout_cross<targets(jj,i-1)) ...
               || (vout_cross<targets(jj,i) && vout_cross>targets(jj,i-1)) ...
           )
                cross_out = cross_out+1;
           end
           if((vin_cross>data_in(1,i) && vin_cross<data_in(1,i)) ...
                   || (vin_cross<data_in(1,i) && vin_cross>data_in(1,i)) ...
                   )
               cross_in = cross_in+1;
           end
        end
%         fprintf(fid, '\n.meas tran delay%d_%d trig V(inp1_1) val=''%e'' cross=%d targ v(outndummy%d) val=''%e'' cross=%d', jj, i, vin_cross,  cross_in, jj, vout_cross, cross_out);        
    end
%     fprintf(fid, '\n.meas delay_out%d PARAM=''(delay1_2', jj);
    for i=3:delay_samples+1
%         fprintf(fid, '+delay%d_%d', jj, i);
    end
%     fprintf(fid, ')/%d''', delay_samples);
    
end

% fprintf(fid, '\n.meas delay PARAM = ''(delay_out1');
% for i=2:Nout
%     fprintf(fid, '+delay_out%d', i);
% end
% fprintf(fid, ')/%d''', Nout);
% 
% fprintf(fid, '\n.meas energy PARAM = ''delay*p_tot''');
  
%  fprintf(fid,'\n.Variation \n.Local_Variation');
%  fprintf(fid,'\n x memristor Rinit=10%%');
%  fprintf(fid,'\n.End_Local_Variation');
%  fprintf(fid,'\n.End_Variation');
 fprintf(fid,'\n.OPTIONS ACCURATE POST=2');
% fprintf(fid,'\n.OPTION METHOD=GEAR'n);
fprintf(fid,'\n.END');
fclose(fid);

command = sprintf('hspice -i %s -o %s.lis -mp %d', filename, fname, num_proc);
tic
system(command);
toc
% fname = strtok(filename,'.');
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