function out = TV(g, H, mu)

[rows cols frames] = size(g);

rho = 2;
f = g;
y1 = zeros(rows, cols, frames);
y2 = zeros(rows, cols, frames);
y3 = zeros(rows, cols, frames);
u1 = zeros(rows, cols, frames);
u2 = zeros(rows, cols, frames);
u3 = zeros(rows, cols, frames);
beta = [1 1 0];

eigHtH      = abs(fftn(H, [rows cols frames])).^2;
eigDtD      = abs(beta(1)*fftn([1 -1],  [rows cols frames])).^2 + abs(beta(2)*fftn([1 -1]', [rows cols frames])).^2;
if frames>1
    d_tmp(1,1,1)= 1; d_tmp(1,1,2)= -1;
    eigEtE  = abs(beta(3)*fftn(d_tmp, [rows cols frames])).^2;
else
    eigEtE = 0;
end
Htg         = imfilter(g, H, 'circular');
[D,Dt]      = defDDt(beta);

[Df1 Df2 Df3] = D(f);
out.relchg = [];
out.objval = [];

rnorm = sqrt(norm(Df1(:))^2 + norm(Df2(:))^2 + norm(Df3(:))^2);

for itr=1:20
    f_old = f;
    rhs   = fftn((mu/rho)*Htg + Dt(u1-(1/rho)*y1,  u2-(1/rho)*y2, u3-(1/rho)*y3));
    eigA  = (mu/rho)*eigHtH + eigDtD + eigEtE;
    f     = real(ifftn(rhs./eigA));
    
    [Df1 Df2 Df3] = D(f);
    v1 = Df1+(1/rho)*y1;
    v2 = Df2+(1/rho)*y2;
    v3 = Df3+(1/rho)*y3;
    v  = sqrt(v1.^2 + v2.^2 + v3.^2);
    v(v==0) = 1;
    v  = max(v - 1/rho, 0)./v;
    u1 = v1.*v;
    u2 = v2.*v;
    u3 = v3.*v;
    y1   = y1 - rho*(u1 - Df1);
    y2   = y2 - rho*(u2 - Df2);
    y3   = y3 - rho*(u3 - Df3);
    

    rnorm_old  = rnorm;
    rnorm      = sqrt(norm(Df1(:)-u1(:), 'fro')^2 + norm(Df2(:)-u2(:), 'fro')^2 + norm(Df3(:)-u3(:), 'fro')^2);
    
    if rnorm>0.7*rnorm_old
        rho  = rho * 2;
    end

    relchg = norm(f(:)-f_old(:))/norm(f_old(:));
    out.relchg(itr) = relchg;

    if relchg < 1e-3;
        break
    end
end

out.f  = f;
out.itr  = itr;
out.y1   = y1;
out.y2   = y2;
out.y3   = y3;
out.rho  = rho;
out.Df1  = Df1;
out.Df2  = Df2;
out.Df3  = Df3;
end

function [D,Dt] = defDDt(beta)
D  = @(U) ForwardD(U, beta);
Dt = @(X,Y,Z) Dive(X,Y,Z, beta);
end

function [Dux,Duy,Duz] = ForwardD(U, beta)
frames = size(U, 3);
Dux = beta(1)*[diff(U,1,2), U(:,1,:) - U(:,end,:)];
Duy = beta(2)*[diff(U,1,1); U(1,:,:) - U(end,:,:)];
Duz(:,:,1:frames-1) = beta(3)*diff(U,1,3); 
Duz(:,:,frames)     = beta(3)*(U(:,:,1) - U(:,:,end));
end

function DtXYZ = Dive(X,Y,Z, beta)
frames = size(X, 3);
DtXYZ = [X(:,end,:) - X(:, 1,:), -diff(X,1,2)];
DtXYZ = beta(1)*DtXYZ + beta(2)*[Y(end,:,:) - Y(1, :,:); -diff(Y,1,1)];
Tmp(:,:,1) = Z(:,:,end) - Z(:,:,1);
Tmp(:,:,2:frames) = -diff(Z,1,3);
DtXYZ = DtXYZ + beta(3)*Tmp;
end