function auc=ROC(result,map,display)
% Calculate the AUC value of result.
% INPUTS:
%   - result:  detection map (rows by columns);
%   - map:     groundtruth (rows by columns);
%   - display: display the ROC curve if display==1.
% OUTPUT:
%   - auc:     AUC values of 'y'.

    [M,N]=size(result);
    p_f=zeros(M*N,1);                    % false alarm rate
    p_d=zeros(M*N,1);                    % detection probability
    [~,ind]=sort(result(:),'descend');
    res=zeros(M*N,1);
    map=reshape(map,M*N,1);
    N_anomaly=sum(map);
    N_pixel=M*N;
    N_miss=0;
    for i=1:M*N                          % calculate pixel by pixel
        res(ind(i))=1;
        N_detected=res'*map;
        if map(ind(i))==0
            N_miss=N_miss+1;
        end
        p_f(i)=N_miss/(N_pixel-N_anomaly);
        p_d(i)=N_detected/N_anomaly;
    end
    auc=trapz(p_f,p_d);                  % calculate the AUC value
    if display==1                        % display the ROC curve if display==1
        figure;plot(p_f,p_d);
    end
end
