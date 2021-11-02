function y=myRGAE(X,SG,lamda,hid)%
% Training of RGAE for hyperspectral anomaly detection
% INPUTS:
%   - X:     HSI data set (rows*columns by bands);
%   - SG:    the Laplacian matrix;
%   - lambda:the tradeoff parameter;
%   - hid:   the number of hidden layer nodes.
% OUTPUT:
%   - y:     final detection map (rows by columns).

    % Parameter settings
    [n,L]=size(X);X=X';
    lr=0.01;epchoes=1200;                   % learning rate and Itermax
    W1=0.01*rand(hid,L);b1=rand(hid,1);     % weights and biases for encoder
    W2=0.01*rand(L,hid);b2=rand(L,1);       % weights and biases for decoder
    batch_num=10;batch_size=n/batch_num;    % training with MBGD
    
    for epcho=1:epchoes
        ind=randperm(n);
        for j=1:batch_num
            x=X(:,ind((j-1)*batch_size+1:j*batch_size));            % fetch the batch and the corresponding Laplacian sub-matrix
            s_G=SG(ind((j-1)*batch_size+1:j*batch_size),ind((j-1)*batch_size+1:j*batch_size));
            % Forward
            z=sigmoid(W1,x,b1);
            x_hat=sigmoid(W2,z,b2);
            res=sqrt(sum((x-x_hat).^2));
            % Backward
            [W1,b1,W2,b2]=trainAE(x,z,x_hat,W1,b1,W2,b2,res,s_G,lr,lamda);
        end
    end
    
    % Output
    Z=sigmoid(W1,X,b1);
    X_hat=sigmoid(W2,Z,b2);
    y=sum((X-X_hat).^2);
end

function y=sigmoid(W,x,b)
% Sigmoid function
    y=1./(1+exp(-(W*x+repmat(b,[1,size(x,2)]))));
end

function [W1_,b1_,W2_,b2_]=trainAE(x,z,x_hat,W1,b1,W2,b2,res,L,lr,lamda)
% Training the network by BP
    [n,num]=size(x);
    D=repmat(1./(2*res),n,1);
    grad=(x_hat-x).*D;
    delta1=grad.*x_hat.*(1-x_hat);
    delta2=z*(L+L');
    tmp=(W2'*delta1+lamda*delta2).*(z.*(1-z));
    W2_=W2-lr*delta1*z'/num;
    b2_=b2-lr*delta1*ones(1,num)'/num;
    W1_=W1-lr*tmp*x'/num;
    b1_=b1-lr*tmp*ones(1,num)'/num;
end
