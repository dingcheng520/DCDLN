function [y nn]=score_binary(pre,rel)
num_data=size(rel,1);
tp=0;tn=0;fp=0;fn=0;
for i = 1:num_data
    if pre(i)<0
        classes=-1;
    elseif pre(i)>=0
        classes=1;
    end
    
    if rel(i)==1 && classes==1  %2是正类，1是反类，col是预测值，rel是真正值
        tp=tp+1;
    elseif rel(i)==-1 && classes==1
        fp=fp+1;
    elseif rel(i)==-1 && classes==-1
        tn=tn+1;
    else
        fn=fn+1;
    end
end
nn=[tp fp tn fn];  %97.22 [35,1, 35, 1]
y1=(tp+tn)*100/num_data; %准确率,accuracy
y2=tp*100/(tp+fp); %精度，Precision
y3=tp*100/(tp+fn); %召回率,Recall
y4=2*y2*y3/(y2+y3); %f1_score
y=[y1,y2,y3,y4];
