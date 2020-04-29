[XTrain, YTrain, XValidation, YValidation, XTest, YTest] = load_data;


imageAugmenterRotation = imageDataAugmenter('RandRotation', [-20,20]);
imageAugmenterXTranslation = imageDataAugmenter('RandXTranslation', [-3 3]);
imageAugmenterYTranslation = imageDataAugmenter('RandYTranslation', [-3 3]);
imageAugmenterXReflexion = imageDataAugmenter('RandXReflection', true);
imageAugmenterYReflexion = imageDataAugmenter('RandYReflection', true);

imageSize = [28 28 1];
augimds = augmentedImageDatastore(imageSize,XTrain,YTrain,...
        'DataAugmentation',imageAugmenterRotation);

num_layers = 1;
num_neurons = 32;

model = get_model(num_layers, imageSize, num_neurons );

epocs_factor = 10;
options = trainingOptions('sgdm', 'MaxEpochs',20*epocs_factor, 'Shuffle',...
        'every-epoch', 'Verbose',true, 'Plots','training-progress', ...
        'ValidationData',{XValidation,YValidation}, ...
        'ValidationFrequency', 20 );
    
net = trainNetwork(augimds,model,options);

YPred = classify(net,XTest);

accuracy = sum(YPred == YTest)/numel(YTest);
