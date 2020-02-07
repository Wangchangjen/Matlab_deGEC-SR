function [MSE,x_hat1,x_hat2] =GLM_quan_FFTBased_cen(x,initial,Nt,Mt,t_max,singular_value, gOut, gxIn,A,L,u,v)
% x: the input signal to compute MSE
% initial: the prior information
% Nt: the dimension of x
% Mt: the dimension of y
% t_max: the number of iteration
% singular_value: the eigenvalue of transform matrix A
% gOut: the posterior information of output for estimation
% gxIn: the posterior information of input for estimation
% A: the transform matrix 
% L: the number of cluster
% u: u matrix of SVD
% v: v matrix of SVD
%% Damping setting
damp_factor = 0.9; 
damp_factor3 = 0.9;
M =Mt/L; % the number of measurement for each cluster
%% Inilization 

DD = zeros(Nt,L);
D1 = zeros(Nt,L);
for iter_A=1:L
    DD(:,iter_A) = singular_value(:,iter_A).^2;
    D1(:,iter_A) = singular_value(:,iter_A);
end

eta1_z=zeros(1,L);
z_hat1=zeros(M,L);
z_hat2=zeros(M,L);
eta2_z=zeros(1,L);
x_hat2=zeros(Nt,L);
eta2_x=zeros(1,L);

% Fienup initialization
xii = abs_initial_projection(initial.abs_y,Mt,Nt,A);

R_2x = repmat(xii,1,L);
V_2x = ones(1,L);


V_1z = zeros(M,L);
V_2z = zeros(M,L);
for iter_A=1:L
V_1z(:,iter_A) =mean(DD(:,iter_A))*ones(1,1)*(Mt/Nt);
V_2z(:,iter_A) = mean(DD(:,iter_A))*ones(1,1)*(Mt/Nt);
end

R_1z = reshape(A*xii,[M L]);
R_2z = reshape(A*xii,[M L]);

D1u = diag(D1)*u'; % constant matrix for each iteration
%%  Iteration
tic
for ii=1:t_max

for iter_A=1:L
    %% 1)Compute the posterior mean and covariance of z
    tic
    [z_hat1(:,iter_A), Q_1_z] = gOut.estim(R_1z(:,iter_A), 1./real(V_1z(:,iter_A)),iter_A);
    eta1_z(:,iter_A) =real(mean(Q_1_z)); %EP-U
    t1=toc;
end

tic
gamma_2_z_new=eta1_z./(1-eta1_z.*V_1z);
r2_z_new=((z_hat1./eta1_z)-(R_1z.*V_1z)).*gamma_2_z_new;
V_2z= damping(V_2z, 1./gamma_2_z_new, damp_factor,ii) ;
R_2z= damping(R_2z, r2_z_new, damp_factor,ii) ;
t2=toc;
%%

tic
for iter_A=1:L
   Vector=(R_2z(:,iter_A).*V_2z(:,iter_A));
   x_hat2(:,iter_A)=v*((V_2x(1,iter_A)./V_2z(1,iter_A)+DD(:,iter_A)).^-1 .* (v'*(R_2x(:,iter_A).*V_2x(1,iter_A))+ D1u*Vector))./V_2z(1,iter_A);
   eta2_x(:,iter_A)=mean((V_2x(1,iter_A)+V_2z(1,iter_A).*DD(:,iter_A)).^-1);
end
t3=toc;
 
tic
gamma_1_x_new = eta2_x./(1-(eta2_x.*V_2x));
r1_x_new=((x_hat2./eta2_x)-(R_2x.*V_2x)).*gamma_1_x_new;
V_1x =  1./gamma_1_x_new;
R_1x = r1_x_new;
t4=toc;
%% MRC combination
tic
    [x_hat1, Q_1_x] = gxIn.estim(R_1x, 1./V_1x) ;  
    Q_1_x=max(Q_1_x,eps);
    eta1_x = mean(real(Q_1_x));
t5=toc;

%----------------------PD_EP----------------------------------

tic
gamma_2_x_new =eta1_x./(1-eta1_x.*V_1x) ;
r2_x_new = ((x_hat1./eta1_x)-(V_1x.*R_1x)).*gamma_2_x_new;
V_2x = damping(V_2x, 1./gamma_2_x_new, damp_factor3,ii) ;
R_2x = damping(R_2x, r2_x_new, damp_factor3,ii) ;
t6=toc;

for iter_A=1:L    
    %% Compute the mean and covariance of z from the linear space
tic    
z_hat2(:,iter_A) = D1u'*((V_2x(1,iter_A)./V_2z(1,iter_A)+DD(:,iter_A)).^-1 .*(v'*(R_2x(:,iter_A).*V_2x(1,iter_A))+ D1u*Vector))/V_2z(1,iter_A);
eta2_z(:,iter_A) = real(sum(DD(:,iter_A)./(V_2z(1,iter_A).*DD(:,iter_A)+V_2x(1,iter_A))))/M;
t7=toc;
end


tic
gamma_1_z_new = eta2_z./(1-eta2_z.*V_2z);
r1_z_new= ((z_hat2./eta2_z)-(R_2z.*V_2z)).*gamma_1_z_new;
V_1z= 1./gamma_1_z_new;
R_1z= r1_z_new; 
t8=toc;   

tt(ii) = t1+t2+t3+t4+t5+t6+t7+t8;
%% MMSE calculation
x_hat1=sign(x_hat1'*x).*x_hat1; % for complex signal
MSE(ii)=norm(x-x_hat1)^(2)/Nt;% NMSE
end
total_iteration_time = mean(tt)*t_max

end
 

