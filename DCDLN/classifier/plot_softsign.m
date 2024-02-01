% x = -2:0.1:2; % 定义 x 的范围
% y = softsign(x); % 计算 softsign 函数值
% plot(x,y,'LineWidth',1.5); % 绘制函数图像
% title('Softsign Function'); % 添加标题
% xlabel('x'); % 添加 x 轴标签
% ylabel('y'); % 添加 y 轴标签

x = -6:0.1:6; % 定义 x 的范围
y = softsign(x); % 计算 softsign 函数值
plot(x,y,'LineWidth',1.5); % 绘制函数图像
title('Softsign Function','FontSize',20); % 添加标题
xlabel('x','FontSize',18); % 添加 x 轴标签
ylabel('y','FontSize',18); % 添加 y 轴标签
ax = gca; % 获取当前坐标轴
ax.XAxisLocation = 'origin'; % 将 x 轴移动到原点
ax.YAxisLocation = 'origin'; % 将 y 轴移动到原点
yticks(-1:1); % 设置 y 轴刻度
ylim([-1.2 1.2]); % 设置 y 轴范围
xticks([-5 0 5]); % 设置 y 轴刻度

% xlim([-10.2 10.2]); % 设置 y 轴范围
box off;

% 
% x = -10:0.1:10; % 定义 x 的范围
% y = softsign(x); % 计算 softsign 函数值
% plot(x,y,'k','LineWidth',1.5); % 绘制函数图像，'k'表示黑色线条
% axis([-10 10 -1.2 1.2]); % 设置坐标轴范围
% axis('equal'); % 设置坐标轴比例相同
% set(gca,'XTick',[],'YTick',[-1 0 1]); % 隐藏 x 轴刻度，设置 y 轴刻度
% box off; % 隐藏周围框
% hold on; % 在同一图中绘制箭头
% quiver(0,0,0.95,0,0,'k','MaxHeadSize',0.8); % 画出 x 轴箭头，9.5表示箭头长度
% quiver(0,0,0,0.95,0,'k','MaxHeadSize',0.8); % 画出 y 轴箭头，0.95表示箭头长度

% x = -10:0.1:10; % 定义 x 的范围
% y = softsign(x); % 计算 softsign 函数值
% plot(x,y,'k'); % 绘制函数图像，'k'表示黑色线条
% axis([-10 10 -1.2 1.2]); % 设置坐标轴范围
% axis('equal'); % 设置坐标轴比例相同
% set(gca,'XTick',-10:10,'YTick',[-1 0 1]); % 设置 x 轴和 y 轴刻度
% box off; % 隐藏周围框