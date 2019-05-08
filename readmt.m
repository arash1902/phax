%============================================================
% Developed by: Mohammad Ansari
% University of Tehran HSpice Toolbox
%============================================================
function data= readmt(filename)
text=fileread(filename);
names=strtok(text,'#');
toTitle=strfind(text,'TITLE');  
hText=text(1:toTitle);
b=textscan(hText,'%s','Delimiter','\n');
headers=length(b{:,:}); %number of header lines
names=(textscan(names,'%s','HeaderLines',headers));  % read the names of variables
clear text hText;
% now read the file
L=length(names{:,:});
size(names{:,:});
fid=fopen(filename);
textscan(fid,'%s',L,'HeaderLines',headers); %scape the variable names
disp('Reading File (.mt# file)')
i=1;
while true
    
    if(feof(fid))
        fclose(fid);
        break;
    end
    f=cell2mat(textscan(fid,'%f',L,'TreatAsEmpty','failed'));
    if(length(f)==L)
        vals2(:,i)=f;
    else
        break;
    end
    i=i+1;
end

data.names=char(names{:}{:});
data.values=vals2;
fclose('all');

end