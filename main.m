%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPLICATION:
%   Hyperspectral Anomaly Detection.
% INPUTS:
%   - data:   HSI data set (rows by columns by bands);
%   - lambda: the tradeoff parameter;
%   - S:      the number of superpixels;
%   - n_hid:  the number of hidden layer nodes.
% OUTPUTS:
%   - y:    final detection map (rows by columns);
%   - AUC:  AUC value of 'y'.
%  REFERENCE:
%   G. Fan, Y. Ma, X. Mei, F. Fan, J. Huang and J. Ma, "Hyperspectral Anomaly
%   Detection With Robust Graph Autoencoders," IEEE Transactions on Geoscience 
%   and Remote Sensing, 2021.
%   G. Fan, Y. Ma, J. Huang, X. Mei and J. Ma, "Robust Graph Autoencoder for 
%   Hyperspectral Anomaly Detection," ICASSP 2021 - 2021 IEEE International 
%   Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, 
%   pp. 1830-1834.
% Written and sorted by Ganghui Fan in 2021. All rights reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;close all;clc
warning('off');

% Data loading and parameter setting
num = 1;
load([num2str(num),'.mat']);
data = (data-min(data(:)))./(max(data(:))-min(data(:)));
lambda=1e-2;S=150;n_hid=100;

% Robust graph autoencoder-based hyperspectral anomaly detector
y = RGAE(data,lambda,S,n_hid);

% Evaluation
y=reshape(y,size(map,1),size(map,2));
AUC=ROC(y,map,0);
