function y=score(pre,rel)

[~,col]=max(pre');  %得出pre中为1的行和列
rel=rel+1;
% y=0;
% for i=1:length(rel)
%     if col(i)==rel(i)
%         y=y+1;
%     end
% end
% y=y/length(rel)*100;  %百分制


num_data=size(rel,1);
tp=0;fp=0;tn=0;fn=0;
for i = 1:num_data
%     %按准确率来算
%     if rel(i)==classes
%         n=n+1;
%     end
    %按真阳性率来算
    if rel(i)==2 && col(i)==2  %2是正类，1是反类，col是预测值，rel是真正值
        tp=tp+1;
    elseif rel(i)==1 && col(i)==2
        fp=fp+1;
    elseif rel(i)==1 && col(i)==1
        tn=tn+1;
    else
        fn=fn+1;
    end
end
y1=(tp+tn)/num_data*100; %ACC,accuracy
y2=tp*100/(tp+fn); %查全率,灵敏度，sensitivity
y3=tn*100/(tn+fp); %特异度,specificity
y4=2*tp*100/(2*tp+fp+fn); %f1_measure
% y=y1*100;  %可以
% y=y2*100;  %不行
% y=y3*100; 
% y=y4*100; 
% y=(y1+y2)/2*100; 
% y=(y1+y3)/2*100; 
% y=(y1+y4)/2*100;
% y=(y2+y3)/2*100; 
% y=(y2+y4)/2*100;  %可以
% y=(y3+y4)/2*100;\
 y= [y1,y2,y3,y4];
% y=2*y1*y3/(y1+y3)*100;  %查全率和查准率的调和平均值
%y=2*y2*y4/(y2+y4)*100;
