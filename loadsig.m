%============================================================
% Developed by: Mohammad Ansari
% University of Tehran HSpice Toolbox
%============================================================
function data= loadsig(filename)

tmptxt=fileread(filename);
s1=sprintf('0.10000E+31\n'); % this number separates monte carlo blocks
s2=sprintf('~'); % the separator char
tmptxt=strrep(tmptxt,s1,s2); %replace the separator number with a separator char ('~')

% in new version of SPICE, 0.10000E+31 is replaced by 0.1000E+031
s1=sprintf('0.1000E+031\n'); % this number separates monte carlo blocks
s2=sprintf('~'); % the separator char
tmptxt=strrep(tmptxt,s1,s2); %replace the separator number with a separator char ('~')


fid=fopen(cat(2,filename,'_loadsig.txt'),'w');
fprintf(fid,'%s',tmptxt);
fclose(fid);
clear tmptxt;
fid=fopen(cat(2,filename,'_loadsig.txt'));
c1=cell2mat(textscan(fid,'%d','HeaderLines',3)); %skeep 3 comment lines and read integers that show the parameter types

num_of_vars = 0;
while(num_of_vars<length(c1))
    c2=textscan(fid,'%s',1,'EndOfLine','&');
    if(~strcmp(c2{1}{1},'') && ~strcmp(c2{1}{1},sprintf('\n')))
        num_of_vars = num_of_vars+1;
        c2_temp{1}{num_of_vars} = strrep(char(c2{1}{1}),sprintf('\n'),'');
    end
end
c2 = c2_temp; % eliminate the blank elements od c2

c3=textscan(fid,'%s',1);%  the separator chars
i=1;
L=length(c1);
typeflag=0; % the flag that determines if  _loadsiftemp.txt file is created or not
switch(char(c3{:,:}))
    
    case {'$&%#'}
        disp('Reading File (Normal File)')
        typeflag=1;
        while true
            vals2=cell2mat(textscan(fid,'%13f',L));
            [y x]=size(vals2);
            if(y==L)
                vals(:,i)=vals2;
                i=i+1;
            else
                break;
            end
            
            
        end
        data.names=char(c2{:}{:});
        fclose(fid);
    case {'MONTE_CARLO'}
        disp('Reading File (Monte carlo File)')
        
        textscan(fid,'%s',1);% scape the separator chars
        
        %L=L+1;
        %c3{:,:};
        
        while true
            
            if(feof(fid))
                fclose(fid);
                break;
            end
            
%             instr=textscan(fid,'%s',1,'EndOfLine',s2,'BufSize',256*1024);
            instr=textscan(fid,'%s',1,'EndOfLine',s2);
            instr=char(instr{:,:});
            fid2=fopen(cat(2,filename,'_loadsigtemp.txt'),'w');
            fprintf(fid2,'%s',instr);
            fclose(fid2);
            fid2=fopen(cat(2,filename,'_loadsigtemp.txt')); %reopen fid2
            monte=cell2mat(textscan(fid2,'%13f',1));% save the monte carlo index
            i=1;
            while true
                
                vals2=cell2mat(textscan(fid2,'%13f',L));
                
                [y x]=size(vals2);
                if(y==L)
                    vals(:,i,monte)=vals2;
                    
                    i=i+1;
                else
                    break;
                    typeflag=1;
                    error('error:L=%d,y=%d',L,y)
                    break;
                end
                
                
            end
            fclose(fid2);
        end
        
        
        data.names=char(c2{:}{:});
        %data.names=char([{char(c2{:}{:});'MONTE_CARLO'}]);
    otherwise
        %if(~isempty(strfind(char(c3{:,:}),':')))
        %if(1)
        disp('Reading File (Trying Nested Sweep)')
        
        ctemp = textscan(fid,'%s',1, 'EndOfLine', '\n');% scape the separator chars
%         ctemp_end=ctemp;
        Lsw = 1;
%         c2{:}{L+Lsw}=char(ctemp{:,:}); % Add the first swept parameter's name to the list
        
        
        % the below lines are added in new version
        
        while(~strcmp(ctemp{:,:},'$&%#'))
            
            c2{:}{L+Lsw}=char(ctemp{:,:}); % Add the other swept parameter's name to the list
            Lsw = Lsw+1;
            ctemp = textscan(fid,'%s',1);% scape the separator chars
        end
         c2{:}{L+Lsw}=char(c3{:,:}); % Add the other swept parameter's name to the list
        

        j=1;
        while true
            
            if(feof(fid))
                fclose(fid);
                break;
            end
            
%             instr=textscan(fid,'%s',1,'EndOfLine',s2,'BufSize',1024*1024);
            instr=textscan(fid,'%s',1,'EndOfLine',s2);
            instr=char(instr{:,:});
            fid2=fopen(cat(2,filename,'_loadsigtemp.txt'),'w');
            fprintf(fid2,'%s',instr);
            fclose(fid2);
            fid2=fopen(cat(2,filename,'_loadsigtemp.txt')); %reopen fid2
            i=1;
%             for i_sw = 1:Lsw
                tempswept=cell2mat(textscan(fid2,'%13f',Lsw));% save the swept parameter's value
                if(~isempty(tempswept))
                    swept(1:Lsw,1,j)=tempswept;
                else
                    if(~feof(fid))
                        typeflag=1;
                        error('Unknown File!')
                    end
                end
%             end
            while true
                
                vals2=cell2mat(textscan(fid2,'%13f',L));
                
                [y x]=size(vals2);
                if(y==L)
                    vals(:,i,j)=vals2;
                    i=i+1;
                else
                    break;
                    %                     if(~feof(fid))
                    %                         disp('Unknown File!')
                    %                         error('error:L=%d,y=%d',L,y)
                    %                         break;
                    %                     end
                end
                
                
            end
            [t tt ttt]=size(vals);
            if(~isempty(tempswept))
                swept(1:Lsw,1:tt,j)=repmat(tempswept,1,tt);
            end
            j=j+1;
            fclose(fid2);
        end
        data.names=char(c2{:}{:});
        vals=cat(1,vals,swept); %add the swept parameter's value to the list
        %else
        
        %end
end


data.values=vals;
%delete temp files
filename=strrep(filename,'/','\');

fclose all;
delstr=sprintf('DEL %s',cat(2,filename,'_loadsig.txt'));
system(delstr);
if(typeflag==0)

    delstr=sprintf('DEL %s',cat(2,filename,'_loadsigtemp.txt'));
    system(delstr);
end


end