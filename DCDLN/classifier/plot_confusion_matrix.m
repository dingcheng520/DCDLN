load Comparation_results.mat
mat1 = [DCDLN.tp,DCDLN.fp;DCDLN.fn,DCDLN.tn;];
% 标签
% label = {'-1','1'};
 
% 混淆矩阵主题颜色
% 可通过各种拾色器获得rgb色值
% maxcolor = [200,35,80]; % 最大值颜色
% mincolor = [230,95,135]; % 最小值颜色

maxcolor = [255,35,255]; % 最大值颜色
mincolor = [255,95,255]; % 最小值颜色
 
% 绘制坐标轴

m = length(mat1);
imagesc(1:m,1:m,mat1)
xticks(1:m)
xlabel('P','fontname','Arial')
xticklabels({'1','-1'})
yticks(1:m)
ylabel('R')
yticklabels({'1','-1'})
set(gca,'xaxislocation','top'); 
set(gca,'FontSize',20)
% 构造渐变色
mymap = [linspace(mincolor(1)/255,maxcolor(1)/255,64)',...
         linspace(mincolor(2)/255,maxcolor(2)/255,64)',...
         linspace(mincolor(3)/255,maxcolor(3)/255,64)'];
    
colormap(mymap)
colorbar()
    xline(1.5, 'Color', 'k');
    yline(1.5, 'Color', 'k'); 
% 色块填充数字
for i = 1:m
    for j = 1:m
        text(i,j,num2str(mat1(i,j)),...
            'horizontalAlignment','center',...
            'verticalAlignment','middle',...
            'fontname','Times New Roman',...
            'fontsize',20);
    end
end
 
% 图像坐标轴等宽
ax = gca;
%ax.FontName = 'Times New Roman';
set(gca,'box','on','xlim',[0.5,m+0.5],'ylim',[0.5,m+0.5]);
axis square
% 色块分界线

% 保存
%saveas(gca,'m.png');