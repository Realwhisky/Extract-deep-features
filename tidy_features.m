
function feat = get_CNNfeatures(im, layers)

global net                                          % ȫ�ֵ���
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
    average=reshape(average,1,1,3);                 % ��ͨ������
end

img = bsxfun(@minus, img, average);                 % ͼ���ֵ��һ��

if enableGPU, img = gpuArray(img); end

res = vl_simplenn(net,img);                         

feat = cell(length(layers), 1);                     % feat �洢�����㼤��ֵ

for ii = 1:length(layers)
    
    if enableGPU
        x = gather(res(layers(ii)).x); 
    else
        x = res(layers(ii)).x;
    end
    
    x = imResample(x, sz_window(1:2));       
                                            
    %   windowing technique sparse
    %   if ~isempty(cos_window),
    %       x = bsxfun(@times, x, cos_window);   % Ȼ������ͼӳ�� �� cos������
    %   end
    
    feat{ii}=x;                              
end
   % feat{3}
end