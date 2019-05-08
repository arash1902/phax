%============================================================
% Developed by: Mohammad Ansari
% University of Tehran HSpice Toolbox
%============================================================
function z=sig(data,name)
%name is a string
%data is spice data struct loaded by ls function

c1={name};
indice=strcmpi(data.names,c1);
s=size(size(data.values)); % to handle 3D matrices in the case of monte carlo or nested sweep
if(s(2)==3) % if monte carlo
    z=data.values(indice,:,:);
else
    z=data.values(indice,:);
end
end