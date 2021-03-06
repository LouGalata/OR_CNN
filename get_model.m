function model = get_model(architecture, imageSize, num_neurons)
    if strcmp(architecture,'v1') == 1
        model = [
        imageInputLayer(imageSize)
%         filterSize = 5; numFilters = 20;
        convolution2dLayer(5,20)
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
%         10 classes - 10 digits
        fullyConnectedLayer(10)
        softmaxLayer
        classificationLayer];
    
    elseif strcmp(architecture,'v1_noBN') == 1
        model = [
            imageInputLayer(imageSize)
    %         filterSize = 5; numFilters = 20;
            convolution2dLayer(5,20)
%             batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2,'Stride',2)
    %         10 classes - 10 digits
            fullyConnectedLayer(10)
            softmaxLayer
            classificationLayer];
    elseif  strcmp(architecture,'v2') == 1
        model = [
            imageInputLayer(imageSize)
%         filterSize = 3; numFilters = 8;
            convolution2dLayer(3,8,'Padding','same')
            batchNormalizationLayer
            reluLayer   

            maxPooling2dLayer(2,'Stride',2)
%         filterSize = 3; numFilters = 16;
            convolution2dLayer(3,16,'Padding','same')
            batchNormalizationLayer
            reluLayer   
            maxPooling2dLayer(2,'Stride',2)
            
 %         filterSize = 3; numFilters = 32;
            convolution2dLayer(3,32,'Padding','same')
            batchNormalizationLayer
            reluLayer   
            maxPooling2dLayer(2,'Stride',2)
 %         filterSize = 3; numFilters = 64;
            convolution2dLayer(3,64,'Padding','same')
            batchNormalizationLayer
            reluLayer  
            maxPooling2dLayer(2,'Stride',2)
            fullyConnectedLayer(10)
            softmaxLayer
            classificationLayer];

    else
        t = strcat(num_layers, ' is a non configured option');
        disp(t)        
end