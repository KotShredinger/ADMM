clear all
close all
clc

%296059
%86016
%219090
%87046
%image
z = im2double(imread('./data/87046.png'));

%matrix
dim = size(z);
h = fspecial('gaussian',[9 9],1);
A = @(z,trans_flag) afun(z,trans_flag,h,dim);
rng(0);

%noies level
noise_level = 10/255;

%observed image
y = A(z(:),'transp') + noise_level*randn(prod(dim),1);
y = min(max(y,0),1);
y = reshape(y,dim);

%ADMM
%lambda = 0.01;
lambda = 0.0005;
%lambda = 0.001;

%initialize variables
dim         = size(y);
N           = dim(1)*dim(2);
v           = 0.5*ones(dim);
x           = v;
u           = zeros(dim);
residual    = inf;
itr = 1;
while(residual>(1e-4)&&itr<=20)
    x_old = x;
    v_old = v;
    u_old = u;
    G = @(z,trans_flag) gfun(z,trans_flag,A,dim);
    xtilde = v-u;
    rhs    = [y(:); xtilde(:)];
    [x,~]  = lsqr(G,rhs,1e-3);
    x      = reshape(x,dim);
    
    %denoising
    vtilde = min(max(x+u,0),1);
    sigma  = sqrt(lambda);
    %v  = TV(vtilde,1,1/sigma^2).f;
    v = RF(vtilde, 3, sigma, sigma, 3);
    %[~,v] = BM3D(1, vtilde, sigma*255);
    
    u = u + (x-v);
    residualx = (1/sqrt(N))*(sqrt(sum(sum((x-x_old).^2))));
    residualv = (1/sqrt(N))*(sqrt(sum(sum((v-v_old).^2))));
    residualu = (1/sqrt(N))*(sqrt(sum(sum((u-u_old).^2))));
    residual = residualx + residualv + residualu;
    itr = itr+1;
end
out = v;


%display
PSNR_output = psnr(out,z);
fprintf('\nPSNR = %3.4f dB \n', PSNR_output);
figure;
subplot(121);
imshow(y);
title('Noise');
%title('Noisy image');


%TV
%RF
%BM3D
subplot(122);
imshow(out);
%title('BM3D');
title('Denoised image');


function y = afun(x,transp_flag,h,dim)
rows = dim(1);
cols = dim(2);
if strcmp(transp_flag,'transp')
    x = reshape(x,[rows,cols]);
    y = imfilter(x,rot90(h,2),'circular');
    y = y(:);
elseif strcmp(transp_flag,'notransp')
    x = reshape(x,[rows,cols]);
    y = imfilter(x,h,'circular');
    y = y(:);
end
end


function y = gfun(x,transp_flag,A,dim)
rows = dim(1);
cols = dim(2);
N    = rows*cols;
if strcmp(transp_flag,'transp')
    x1   = x(1:N);
    x2   = x(N+1:2*N);
    Atx  = A(x1,'transp');
    y    = Atx + x2;
elseif strcmp(transp_flag,'notransp')
    Ax   = A(x,'notransp');
    y    = [Ax; x];
end
end