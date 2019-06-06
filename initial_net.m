function initial_net()

global net;
net = load(fullfile('model', 'imagenet-vgg-verydeep-19.mat'));

% �Ƴ�ȫ���Ӳ������㣨softmax�㣩
net.layers(37+1:end) = [];

% GPUģʽ
global enableGPU;
if enableGPU
    net = vl_simplenn_move(net, 'gpu');
end

net=vl_simplenn_tidy(net);

end