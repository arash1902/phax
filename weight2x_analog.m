function weight2x_analog( weights, biases, filename)
% m is the number of layers
% weights: 1xN-1 cell array, cell i containg the weights from layer(i) to
% layer(i+1)
% biases: 1xN-1 array containing the bias of each layer

min_conductance =1./8.33e6 ;
max_conductance =1./127e3;
fname = strtok(filename,'.');
for ii=1:length(weights) % for each layer
    fnm = sprintf('init_%s_%d.txt', fname, ii);
    fid = fopen(fnm, 'w+');
    [Nout, Nin] = size(weights{ii});
    Nin= Nin/2;
    for k=1:Nout
        w = weights{ii}(k,:);
        b = biases{ii}(k,:); 
        
        % note that each output can have its own scaling factor
%         scaling = min_conductance./min(min(w))./sum(sum(abs(w)));
%         w = w*scaling;
%         b = b*scaling;
        V_PLUS_MINUS = 0.5*2; % *2 is because Arash has trained the net with +-0.5 biases
        W = [w,b/V_PLUS_MINUS];
        % NOTE THAT b IS DEVIDED BY V_PLUS_MINUS BECAUSE OF ITS UNITY EFFECT
        % IN THE FINAL RESULT

        scaling = min_conductance./min(min(abs(W)));
%   scaling = max_conductance./max(max(abs(W)));
%         sig_mid = sig_m0./2*ones(length(W),1);
        sig_p = W(:,1:2:end)*scaling;
        sig_n = W(:,2:2:end)*scaling;
        if(sum(sig_p<0)>0 || sum(sig_n<0)>0)
            sig_p
            sig_n
               error('Negative resistance produced');
        end
        Lw = length(w)/2;
            bp = sig_p(Lw+1:end);
            bn = sig_n(Lw+1:end);
        if(ii==1)
            wp = sig_p(1:Lw);  
            wn = sig_n(1:Lw);
        else
            wn = sig_p(1:Lw);  
            wp = sig_n(1:Lw);
        end
        % modify if the sigma is negative
        
%          for jj=1:Nin
%             fprintf(fid, '\n .param si%d_%d_p = %3.2e ', jj, k, wp(jj));
%             fprintf(fid, '\n .param si%d_%d_n = %3.2e ', jj, k, wn(jj));
%         end
        % print the bias memristor values
%           fprintf(fid, '\n .param sib1_%d = %3.2e', k(1), double(bp(1)+eps));
%          fprintf(fid, '\n .param sib0_%d = %3.2e', k(1), double(bn(1)+eps)); 
        
        % print the crossbar memristor values
        % convert conductance to resistance
        wp = r2x( 1./wp, 1, 1.6e-4, 1.6e-4, 0.05, 0.985, 0.985);
        wn = r2x( 1./wn, 1, 1.6e-4, 1.6e-4, 0.05, 0.985, 0.985);
        bp = r2x( 1./bp, 1, 1.6e-4, 1.6e-4, 0.05, 0.985, 0.985);
        bn = r2x( 1./bn, 1, 1.6e-4, 1.6e-4, 0.05, 0.985, 0.985);
%         wp = 1./wp;
%         wn = 1./wn;
%         bp = 1./bp;
%         bn = 1./bn;

        for jj=1:Nin
            fprintf(fid, '\n .param ri%d_%d_p = %7.6e ', jj, k, wp(jj));
            fprintf(fid, '\n .param ri%d_%d_n = %7.6e ', jj, k, wn(jj));
%============================= for MONTE CARLO SIM. use the following two
%line:
%             fprintf(fid, '\n .param ri%d_%d_p = gauss(%3.2e,0.1,3) ', jj, k, wp(jj));
%             fprintf(fid, '\n .param ri%d_%d_n = gauss(%3.2e,0.1,3) ', jj, k, wn(jj));
%===============================================================       
        end
        % print the bias memristor values
         fprintf(fid, '\n .param rib1_%d = %7.6e', k, double(bp(1)+eps));
         fprintf(fid, '\n .param rib0_%d = %7.6e', k, double(bn(1)+eps)); 
    end
    fprintf(fid, '\r\n');
    fclose(fid);
end
end