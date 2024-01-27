%ds = datastore(fullfile("E:\Logos"), "IncludeSubfolders", true, "Type", "image", "FileExtensions", ".jpg", 'LabelSource', 'foldernames');

[dsTrain, dsValidation] = splitEachLabel(ds,0.9,'randomized');

numClasses = numel(categories(dsTrain.Labels));

layers = [
    imageInputLayer([227 227 3])
    
    convolution2dLayer(5,16,'Padding','same')
    batchNormalizationLayer
    reluLayer('Name','relu_1')
    
    convolution2dLayer(3,32,'Padding','same','Stride',2)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    additionLayer(2,'Name','add')
    
    averagePooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

lgraph = layerGraph(layers);

skipConv = convolution2dLayer(1,32,'Stride',2,'Name','skipConv');
lgraph = addLayers(lgraph,skipConv);

lgraph = connectLayers(lgraph,'relu_1','skipConv');
lgraph = connectLayers(lgraph,'skipConv','add/in2');

options = trainingOptions('sgdm', ...
    'MaxEpochs',8, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

inputSize = [227 227 3];

imageAugmenter = imageDataAugmenter('RandXTranslation', [-20 20], 'RandYTranslation', [-20 20], 'RandRotation', [-180 180], 'RandScale', [0.5 2]);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), dsTrain, 'DataAugmentation', imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2), dsValidation);

net = trainNetwork(augimdsTrain,lgraph,options);

[YPred,scores] = classify(netTransfer, augimdsValidation);

YValidation = dsValidation.Labels;
accuracy = mean(YPred == YValidation);