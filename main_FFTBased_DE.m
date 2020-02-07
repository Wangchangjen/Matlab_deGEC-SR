% main_function_deGEC-SR
clc
clear all
%% parameter setting 
clc
SNR_dB =10;%  SNR(dB)
N=100; % dimension of transmitted signal x
M=4*N ; % dimension of observation y
lamda=0.5; %sparasity
tr=1;% number of realizations
t_max=50; %number of iteration
L=10; % number of decentralized clusters
x_type = 0; % Types of intput x, 0: the intput signal is bernoulli complex-valued-AWGN, 1: the input signal is bernoulli real-valued-AWGN

%% Nonlinear-function for Q(.)
B_Bit='abs';
%% Decentralized groups
M_L = M/L; % the measurement number of each cluster
cluster_tab = [1:M_L:M;M_L:M_L:M].'; % the index for each cluster

for jj=1:tr
    %% input-x
    if x_type == 0
        gxIn = CAwgnEstimIn(0,1/lamda, lamda);
        bernoulli=rand(N,1)>(1-lamda);% bernoulli distribution
        Gauss=sqrt(1/(2*lamda))*(randn(N,1)+1i*randn(N,1));% complex  Gaussian  distribution
        x=(bernoulli.*Gauss);%  input x  
    else
        gxIn = AwgnEstimIn(0,1/lamda, lamda);
        bernoulli=rand(N,1)>(1-lamda);
        Gauss=sqrt(1/(lamda))*(randn(N,1));
        x=(bernoulli.*Gauss);
    end
   %% transform matrix-A
        A=crandn(M,N)/sqrt(2*M);
        z=A*x;


   %% SVD for each cluster
    for c =1:L 
        if M_L <= N
            tic
            [u d v]=svd(A(1+(c-1)*M_L:c*M_L,:));  
            U(c)=UnitaryOperator(u);
            V(c)=UnitaryOperator(v');
            D(c)=UnitaryOperator(d);
            svd_eig(:,c) = diag(d);
            t_SVD(c)=toc;
        else
            tic
            [u d v]=svd(A(1+(c-1)*M_L:c*M_L,:),'econ');  
            U(c)=UnitaryOperator(u);
            V(c)=UnitaryOperator(v');
            svd_eig(:,c) = diag(d);
            D(c)=UnitaryOperator(d);
            t_SVD=toc
        end
    end
    
    %% noise 
    if B_Bit == 'abs'
        sigma2 = 10^(-0.1*SNR_dB)*mean(abs(z).^2);
    end
    noise=sqrt(sigma2/2)*(randn(M,1)+1i*randn(M,1));% noise(complex Gaussian distribution)
    y = z + noise; % observation

    %% Nonlinear estimation function
    
    if (B_Bit == 'abs')
        initial.abs_y = abs(y);
        gOut=ncCAwgnEstimOut(initial.abs_y, sigma2*ones(M_L,1),cluster_tab);
        indx_big = find( abs(x) > 0.1*sqrt(mean(abs(x).^2)) );
        initial.xmean0 = mean(x(indx_big));
        initial.xvar0 = var(x(indx_big)); 
        initial.lamda =lamda;
    end        

%% Run deGEC-SR (L > 1) or GEC-SR (L=1)
    if L ~=1
        [MSE_mtx_GLM(:,jj),x_est3,x_hat3]...
           = GLM_quan_FFTBased_decen(x,initial,N,M,t_max, svd_eig, gOut, gxIn,A,L,U,V,D); % run GLM
    else
        [MSE_mtx_GLM(:,jj),x_est3,x_hat3]...
           = GLM_quan_FFTBased_cen(x,initial,N,M,t_max, svd_eig, gOut, gxIn,A,L,u,v); % run GLM
    end

end

MSE_GLM=sum(MSE_mtx_GLM,2)/tr; % average over tr realization
plot(1:t_max,10*log10(MSE_GLM),'bo-');hold on % dB



