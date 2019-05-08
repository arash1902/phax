function make_layer_inv_diff(layer_num, Nin, Nout , filename,  sz, opt_imp)
% optional implementation:
% opt_imp = 1: memristor with nominal value
% opt_imp = 2; memristor with gaussian distribution
% opt_imp = 3; resistor with nominal value
%=====
% sz= inverter sizing

% opt_imp = 1;
fid = fopen(filename, 'a');
fname = strtok(filename,'.');
fprintf(fid, '\n\n\n*=======================layer%d=======================',layer_num);
fprintf(fid, '\n.subckt layer%d',layer_num);
fprintf(fid, ' b0 b1');
for i=1:Nin
    fprintf(fid, ' xp%d', i);
    fprintf(fid, ' xn%d', i);
end
for j=1:Nout
    fprintf(fid, ' outp%d', j);
    fprintf(fid, ' outn%d', j);
end
fprintf(fid, ' vdd_logic vss');
fprintf(fid, '\n.param vtpn = ''1.1''\n' );
fprintf(fid, '.prot\n.include ''comp.sp''\n.include ''init_%s_%d.txt''\n.unprot\n',fname, layer_num);
%print the crossbar
for row=1:Nin
    for col = 1:Nout
        %  =============================== BIAS =====================
        
        switch rem(opt_imp,3)
            case 1
                %  ============= option 1: Nominal Value with Memmristor:
                fprintf(fid, '\nxm%d_%d_p xp%d	in%d	w%d_%d_p	0 memristor vtp=''vtpn'' vtn=''vtpn''	xo=''ri%d_%d_p''', row,col,row,  col, row, col, row, col);
                fprintf(fid, '\nxm%d_%d_n xn%d	in%d	w%d_%d_n	0	memristor vtp=''vtpn'' vtn=''vtpn''	xo=''ri%d_%d_n''', row,col,row,  col, row, col, row, col);
                
            case 2
                %  ============= option 2: Gaussina Value with Memmristor:
                fprintf(fid, '\nxm%d_%d_p xp%d	in%d	w%d_%d_p	0 memristor vtp=''vtpn'' vtn=''vtpn''	Rinit=''gauss(ri%d_%d_p,0.1,3)''', row,col,row,  col, row, col, row, col);
                fprintf(fid, '\nxm%d_%d_n xn%d	in%d	w%d_%d_n	0	memristor vtp=''vtpn'' vtn=''vtpn''	Rinit=''gauss(ri%d_%d_n,0.1,3)''', row,col,row,  col, row, col, row, col);
                
            case 3
                %  ============= option 3: Nominal Value with Resistor:
                fprintf(fid, '\nrm%d_%d_p xp%d	in%d  ''ri%d_%d_p''', row,col,row,  col, row, col);
                fprintf(fid, '\nrm%d_%d_n xn%d	in%d  ''ri%d_%d_n''', row,col,row,  col, row, col);
            otherwise
                error('Improper opt_imp');
        end
    end
    
end
% print bias weights
for col = 1:Nout
    %  =============================== Weights =====================
    
    switch rem(opt_imp,3)
        case 1
            %  ============= option 1: Nominal Value with Memmristor:
            fprintf(fid, '\nxb0_%d b0	in%d	wb0_%d	0	memristor vtp=''vtpn'' vtn=''vtpn''	xo=''rib0_%d''', col,  col, col, col);
            fprintf(fid, '\nxb1_%d b1	in%d	wb1_%d	0	memristor vtp=''vtpn'' vtn=''vtpn''	xo=''rib1_%d''', col,  col, col, col);
            
        case 2
            %  ============= option 2: Gaussina Value with Memmristor:
            fprintf(fid, '\nxb0_%d b0	in%d	wb0_%d	0	memristor vtp=''vtpn'' vtn=''vtpn''	Rinit=''gauss(rib0_%d,0.1,3)''', col,  col, col, col);
            fprintf(fid, '\nxb1_%d b1	in%d	wb1_%d	0	memristor vtp=''vtpn'' vtn=''vtpn''	Rinit=''gauss(rib1_%d,0.1,3)''', col,  col, col, col);
            
        case 3
            %  ============= option 3: Nominal Value with Resistor:
            fprintf(fid, '\nrb0_%d b0	in%d ''rib0_%d''', col,  col, col);
            fprintf(fid, '\nrb1_%d b1	in%d ''rib1_%d''', col,  col, col);
        otherwise
            error('Improper opt_imp');
    end
    % ======================== 2 inverters =====================================
    if(opt_imp==4)
         %short circuit
        fprintf(fid, '\nvcndummy%d	in%d	outn%d	 0\n', col, col ,col); %short circuit
        fprintf(fid, '\nvcpdummy%d	outn%d	outp%d	 0\n', col, col ,col);
    else
        fprintf(fid, '\nxcn%d	in%d	outn%d	 vdd_logic vss inv_diff sz=%d\n', col, col ,col, sz );
        fprintf(fid, '\nxcp%d	outn%d	outp%d	 vdd_logic vss inv_diff sz=%d\n', col, col ,col, sz );
    end
    % ======================== 3 inverters =====================================
    %          fprintf(fid, '\nxcpp%d	in%d	outx%d	 vdd_logic vss inv_diff sz=%d\n', col, col ,col, sz );
    %          fprintf(fid, '\nxcp%d	outx%d	outp%d	 vdd_logic vss inv_diff sz=&d\n', col, col ,col, sz );
    %          fprintf(fid, '\nxcn%d	outp%d	outn%d	 vdd_logic vss inv_diff sz=%d\n', col, col ,col, sz );
    
    % ======================== 4 inverters =====================================
    %     fprintf(fid, '\nxcpp%d	in%d	outbb%d	 vdd_logic vss inv_diff sz=%d\n', col, col ,col, sz );
    %     fprintf(fid, '\nxcpn%d	outbb%d	outx%d	 vdd_logic vss inv_diff sz=%d\n', col, col ,col ,sz );
    %     fprintf(fid, '\nxcp%d	outx%d	outn%d	 vdd_logic vss inv_diff sz=%d\n', col, col ,col, sz );
    %     fprintf(fid, '\nxcn%d	outn%d	outp%d	 vdd_logic vss inv_diff sz=%d\n', col, col ,col, sz );
end

%fprintf(fid, '\nvdd vdd 0 	''vdd_train''\n');
%fprintf(fid, '\n.include ''inputs_%s.txt''\n',fname);
fprintf(fid, '\n.ends');
fclose(fid);
end