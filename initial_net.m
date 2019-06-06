function initial_net()

global net;
net = load(fullfile('model', 'imagenet-vgg-verydeep-19.mat'));

% 移除全连接层与分类层（softmax层）
net.layers(37+1:end) = [];

% GPU模式
global enableGPU;
if enableGPU
    net = vl_simplenn_move(net, 'gpu');
end

net=vl_simplenn_tidy(net);

end