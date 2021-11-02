function [SG,X_new,idex]=SuperGraph(data,S)
% Construction of Laplacian matrix with SuperGraph
% INPUTS:
%   - data:  HSI data set (rows by columns by bands);
%   - S:     the number of superpixels.
% OUTPUT:
%   - SG:    the Laplacian matrix;
%   - X_new: the shuffled data set;
%   - idex:  the indices of the rows of 'X_new' corresponding to the reshaped 'data'.

    % Apply superpixel segmentation on compressed 'data'
    [M,N,L]=size(data);
    X=reshape(data,M*N,L);
    Y=myPCA(data);
    y=Y(:,:,end);
    y=(y-min(y(:)))./(max(y(:))-min(y(:)));
    [labels,nums]=superpixels(y,S);
    
    % Construct the Laplacian matrix with modified method
    W=sparse(M*N,M*N);
    spec2=4;X_new=[];
    idex=[];
    cnt=0;
    for num=1:nums
        idx=find(labels==num);
        K=size(idx,1);
        x=X(idx,:);
        X_new=[X_new;x];
        idex=[idex;idx];
        for i=1:K
            s=x(i,:);
            tmp=zeros(K);
            for j=i+1:K
                tmp(i,j)=exp(-sum((s-x(j,:)).^2,2)/(2*spec2));
            end
        end
        W(cnt+1:cnt+K,cnt+1:cnt+K)=tmp+tmp';
        cnt=cnt+K;
    end
    SG=diag(sum(W))-W;
end

