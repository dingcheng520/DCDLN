function [coeff,element]=PCA_extra(x)
% %主成分分析,前99%的元素
% ss=zscore(x);
% %0均值标准化
% [ss,PS] = mapstd(x',0,1);
% ss=ss';
% X = mapstd('apply',X',PS)';

[coeff,score,latent,tsquare]=pca(x);
contri=(100*latent/sum(latent));
for element=1:size(contri,1)
    if sum(contri(1:element))>80
        break;
    end
end
% xapp=ss*coeff(:,1:element);