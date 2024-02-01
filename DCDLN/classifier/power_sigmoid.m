function y=power_sigmoid(x)
r=2;n=4;
for row = 1:1:size(x,1)
    for column = 1:1:size(x,2)
         a= x(row,column);
        if abs(a) > 1
            y(row,column) = a^(2 * r - 1);
        else 
            y(row,column) = ((1 + exp(-n)) / (1 - exp(-n))) * (1 - exp(-n * a)) / (1 + exp(-n * a));
        end
    end
end   