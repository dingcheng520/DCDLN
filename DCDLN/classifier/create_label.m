function label = create_label(Y)
%输入：限定Y的最小值为1，最大值为类别数，向量形式而非矩阵
class=max(Y);
numberpclass=length(Y); 
label=Y;            %lable是标签
for i=1:numberpclass                   
    if label(1,i) == 0
        label(1,i) = -1;
    end%按照Y来打标签
end