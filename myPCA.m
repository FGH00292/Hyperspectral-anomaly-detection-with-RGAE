function Y=myPCA(data)
% PCA
    [M,N,L]=size(data);
    X=reshape(data,M*N,L);
    sigma=X'*X;
    [V,~]=eig(sigma);
    Y=X*V;
    Y=reshape(Y,M,N,L);
end
