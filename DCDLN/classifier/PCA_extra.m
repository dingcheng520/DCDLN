function [coeff,element]=PCA_extra(x)
% %���ɷַ���,ǰ99%��Ԫ��
% ss=zscore(x);
% %0��ֵ��׼��
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