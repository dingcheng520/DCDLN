% x = -2:0.1:2; % 定义 x 的范围
% y = power_sigmoid(x); % 计算 softsign 函数值
% plot(x,y,'LineWidth',1.5); % 绘制函数图像
% title('Power_sigmoid Function'); % 添加标题
% xlabel('x'); % 添加 x 轴标签
% ylabel('y'); % 添加 y 轴标签




% x = -1.5:0.1:1.5; % 定义 x 的范围
% y = power_sigmoid(x); % 计算 softsign 函数值
% plot(x,y,'LineWidth',1.5); % 绘制函数图像
% title('Power sigmoid','FontSize',20); % 添加标题
% xlabel('x','FontSize',18); % 添加 x 轴标签
% ylabel('y','FontSize',18); % 添加 y 轴标签
% lgd = legend('r=2,n=4','Location','northeast');
% set(lgd,'FontSize',14); % 设置图例字体大小
% ax = gca; % 获取当前坐标轴
% ax.XAxisLocation = 'origin'; % 将 x 轴移动到原点
% ax.YAxisLocation = 'origin'; % 将 y 轴移动到原点
% yticks(-1:1); % 设置 y 轴刻度
% ylim([-2 2]); % 设置 y 轴范围
% xticks([-1 0 1]); % 设置 y 轴刻度
% axis('equal');
% % xlim([-10.2 10.2]); % 设置 y 轴范围
% box off;


% x = -1.5:0.1:1.5; % 定义 x 的范围
% y = cos(x);
% plot(x,y)


% 创建数据
x = linspace(-10, 10, 100);
y = sin(x);

% 绘制图像
figure;
plot(x, y, 'b-', 'LineWidth', 2);

% 设置坐标轴范围
axis([-10 10 -1.5 1.5]);

% 添加箭头到x轴
annotation('arrow', [0.5 0.5], [0.5 0.85], 'HeadStyle', 'vback2', 'HeadWidth', 8, 'LineWidth', 1.5);
% 添加箭头到y轴
annotation('arrow', [0.5 0.15], [0.5 0.5], 'HeadStyle', 'vback2', 'HeadWidth', 8, 'LineWidth', 1.5);

% 添加标签
xlabel('x');
ylabel('y');

% 图形美化
grid on;
box on;