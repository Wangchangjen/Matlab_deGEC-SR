function [MSE,x_hat1,x_hat2] =DeGEC_SR(x,initial,Nt,Mt,t_max,singular_value, gOut, gxIn,A,L,U,V,D)
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
% U: u matrix of SVD and its multiplication operation
% V: v matrix of SVD and its multiplication operation
% D; eigenvalue matrix of SVD and its multiplication operation
%% Damping setting
damp_factor = 0.9; 
damp_factor3 = 0.9;
M =Mt/L; % the number of measurement for each cluster
%% Inilization 

DD = zeros(M,L);
D1 = zeros(M,L);
for iter_A=1:L
    DD(:,iter_A) = singular_value(:,iter_A).^2;
    D1(:,iter_A) = singular_value(:,iter_A);
end


% Fienup initialization
xii = abs_initial_projection(initial.abs_y,Mt,Nt,A);

R_2x = repmat(xii,1,L);
V_2x = ones(1,L);


for iter_A=1:L
V_1z(:,iter_A) =mean(DD(:,iter_A))*ones(1,1)*(Mt/Nt);
V_2z(:,iter_A) = mean(DD(:,iter_A))*ones(1,1)*(Mt/Nt);
end

R_1z = reshape(A*xii,[M L]);
R_2z = reshape(A*xii,[M L]);

for iter_A=1:L
D1u(:,:,iter_A) = U(iter_A).Opt(D(iter_A).HaarMatrix);
DD0(:,iter_A) = [DD(:,iter_A);zeros(Nt-M,1)];
end
%%  Iteration
tic
for ii=1:t_max
%%  Each-cluster
tic
for iter_A=1:L
    %% 1)Compute the posterior mean and covariance of z
    
    [z_hat1(:,iter_A), Q_1_z] = gOut.estim(R_1z(:,iter_A), 1./real(V_1z(:,iter_A)),iter_A);
    eta1_z(:,iter_A) =real(mean(Q_1_z)); %EP-U
    
end
t1=toc;

tic
gamma_2_z_new = eta1_z./(1-eta1_z.*V_1z);
r2_z_new=bsxfun(@times,bsxfun(@minus,bsxfun(@rdivide,z_hat1,eta1_z), bsxfun(@times,V_1z,R_1z)),gamma_2_z_new);
V_2z= damping(V_2z, 1./gamma_2_z_new, damp_factor,ii) ;
R_2z= damping(R_2z, r2_z_new, damp_factor,ii) ;
t2=toc;

tic
for iter_A=1:L
   Vector(:,iter_A) = R_2z(:,iter_A).*V_2z(:,iter_A);
   x_hat2(:,iter_A)=V(iter_A).tOpt((V_2x(1,iter_A)./V_2z(1,iter_A)+DD0(:,iter_A)).^-1 .* ...
       (V(iter_A).Opt(R_2x(:,iter_A).*V_2x(1,iter_A))+ D1u(:,:,iter_A)'*Vector(:,iter_A)))./V_2z(1,iter_A);   
   
   eta2_x(:,iter_A)=mean((V_2x(1,iter_A)+V_2z(1,iter_A).*DD0(:,iter_A)).^-1);
end
t3=toc;

tic
gamma_1_x_new = (1./eta2_x-V_2x).^-1;
r1_x_new=bsxfun(@times,bsxfun(@minus,bsxfun(@rdivide,x_hat2,eta2_x), bsxfun(@times,V_2x,R_2x)),gamma_1_x_new);
V_1x = 1./gamma_1_x_new;
R_1x = r1_x_new;
t4=toc;

%% Fusion_center
tic
    Fusion_v = 1./real(sum(V_1x,2));
    Fusion_x = bsxfun(@times,Fusion_v,sum(bsxfun(@times,R_1x,V_1x),2));
    [x_hat1, Q_1_x] = gxIn.estim(Fusion_x, Fusion_v) ;
    Q_1_x=max(Q_1_x,eps);
    eta1_x = mean(real(Q_1_x)); 
t5=toc;

%%  Each-cluster
tic
gamma_2_x_new =eta1_x./(1-(eta1_x.*V_1x)) ;
r2_x_new = bsxfun(@times, bsxfun(@minus,x_hat1./eta1_x,bsxfun(@times,V_1x,R_1x)) ,gamma_2_x_new);
V_2x = damping(V_2x, 1./gamma_2_x_new, damp_factor3,ii) ;
R_2x = damping(R_2x, r2_x_new, damp_factor3,ii) ;
t6=toc;

tic
for iter_A=1:L   
    %% Compute the mean and covariance of z from the linear space
z_hat2(:,iter_A) =  D1u(:,:,iter_A)*((V_2x(1,iter_A)./V_2z(1,iter_A)+DD0(:,iter_A)).^-1 .* ...
       (V(iter_A).Opt(R_2x(:,iter_A).*V_2x(1,iter_A))+ D1u(:,:,iter_A)'*Vector(:,iter_A))./V_2z(1,iter_A));   
eta2_z(:,iter_A) = real(sum(DD(:,iter_A)./(V_2z(1,iter_A).*DD(:,iter_A)+V_2x(1,iter_A))))/M;
end
t7=toc;

tic
gamma_1_z_new = eta2_z./(1-eta2_z.*V_2z);
r1_z_new=bsxfun(@times,bsxfun(@minus,bsxfun(@rdivide,z_hat2,eta2_z), bsxfun(@times,V_2z,R_2z)),gamma_1_z_new);
V_1z = 1./gamma_1_z_new;
R_1z = r1_z_new;
t8=toc;
%% MMSE calculation
x_hat1=sign(x_hat1'*x).*x_hat1; % for complex signal
MSE(ii)=norm(x-x_hat1)^(2)/Nt;% MSE

% Except fusion center (t5), we average the sum time because the computation is parallel in each cluster. (t1~t8)
tt(ii) = (t1+t2+t3+t4+t6+t7+t8)/L +t5;
end
total_iteration_time= mean(tt)*t_max
%   MRCvx_data(:,ii)=MRCvx;

end
 

