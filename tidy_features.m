
function feat = get_CNNfeatures(im, layers)

global net                                          % 全局调用
global enableGPU         

if isempty(net)          
   initial_net();        
end

sz_window = size(im);

img = single(im);        

img = imResample(img, net.meta.normalization.imageSize(1:2));

average=net.meta.normalization.averageImage;        % batch normal ~

% *************************************************************************
if numel(average)==3                                
    average=reshape(average,1,1,3);                 % 三通道输入
end

img = bsxfun(@minus, img, average);                 % 图像均值归一化

if enableGPU, img = gpuArray(img); end

res = vl_simplenn(net,img);                         

feat = cell(length(layers), 1);                     % feat 存储特征层激活值

for ii = 1:length(layers)
    
    if enableGPU
        x = gather(res(layers(ii)).x); 
    else
        x = res(layers(ii)).x;
    end
    
    x = imResample(x, sz_window(1:2));       
                                            
    %   windowing technique sparse
    %   if ~isempty(cos_window),
    %       x = bsxfun(@times, x, cos_window);   % 然后特征图映射 作 cos窗处理
    %   end
    
    feat{ii}=x;                              
end
   % feat{3}
end