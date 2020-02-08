function [xii] =abs_initial_projection(abs_y,Mt,Nt,A)
yy = abs_y.^2;
delta = Mt/Nt;
ymean = mean(yy);
yy = yy/ymean;
yplus = max(yy,0);
T = (yplus-1)./(yplus+sqrt(delta)-1); 
T = T*ymean;
Yfunc = @(a) 1/Mt*A'*(T.*(A*a));
opts.isreal = false;
[xii,~] = eigs(Yfunc, Nt, 1, 'lr', opts);
uu = abs(A*xii).*abs_y;
l = abs(A*xii).*abs(A*xii);
s = norm(uu(:))/norm(l(:));
xii = xii*s;
for iter_inial=1:1:3
    z_ob = A*xii;
    z_abs = abs_y.*sign(z_ob);
    xii = pinv(A)*z_abs;
end
end