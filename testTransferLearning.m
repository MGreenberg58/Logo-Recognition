ds = datastore(fullfile("E:\Logos\LogoDet-3K\Transportation\"), "IncludeSubfolders", true, "Type", "image", "FileExtensions", ".jpg", 'LabelSource', 'foldernames');

[dsTrain, dsValidation] = splitEachLabel(ds,0.9,'randomized');

net = resnet101;
analyzeNetwork(net);
lgraph = layerGraph(net);

inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(dsTrain.Labels));

lgraph = removeLayers(lgraph, 'fc1000');
lgraph = removeLayers(lgraph, 'prob');
lgraph = removeLayers(lgraph, 'ClassificationLayer_predictions');

layers = [
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',15,'BiasLearnRateFactor',15,"Name",'New_Fully_Connected')
    softmaxLayer('Name','New_Softmax_Layer')
    classificationLayer('Name','New_Classification')];

lgraph = addLayers(lgraph, layers);

lgraph = connectLayers(lgraph,'pool5','New_Fully_Connected');


%analyzeNetwork(lgraph)

imageAugmenter = imageDataAugmenter('RandXTranslation', [-20 20], 'RandYTranslation', [-20 20], 'RandRotation', [-180 180], 'RandScale', [0.5 2]);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), dsTrain, 'DataAugmentation', imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2), dsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots','training-progress', ...
    'OutputNetwork', 'last-iteration');

%netTransfer = trainNetwork(augimdsTrain, lgraph, options);

[YPred,scores] = classify(netTransfer, augimdsValidation);

YValidation = dsValidation.Labels;
accuracy = mean(YPred == YValidation);
