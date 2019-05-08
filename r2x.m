% function [x, xx, xxx]= r2x( res, v, max_v, a1, a2, b)
function xxx= r2x( res, max_v, a1, a2, b, xp, xn)
    %coeff = 1.042e-5;
    coeff = (b^3)*(max_v^2)/12;
%     i = a1*sinh(b*v);n
%     ii = a1*(b+((b^3)*v^2)/6);
    iii = a1*(b + coeff);
%     x = v / (res*i);
%     xx = 1/(res*ii);
    xxx = 1./(res.*iii);
    if(sum(xxx>xp)>0 || sum(xxx<1-xn)>0)
        error('Out of range state variable created: x=%e',xxx)
    end
end