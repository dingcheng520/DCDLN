function label = create_label(Y)
%���룺�޶�Y����СֵΪ1�����ֵΪ�������������ʽ���Ǿ���
class=max(Y);
numberpclass=length(Y); 
label=Y;            %lable�Ǳ�ǩ
for i=1:numberpclass                   
    if label(1,i) == 0
        label(1,i) = -1;
    end%����Y�����ǩ
end