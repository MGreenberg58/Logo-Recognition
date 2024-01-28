ds = imageDatastore("LogoDet-3K (1)/LogoDet-3K/Transportation/", ...
    FileExtensions=[".jpg"], ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");
nImages = numel(ds.Files);
images = ds.Files;
labels = ds.Labels;

imagepaths = strings(nImages,1);
for i = 1:nImages
    imagepaths(i) = string(images(i));
end

%% Generate training, validation, and test sets
rng("default");
shuffledIndices = randperm(nImages);
idx = floor(0.6 * length(shuffledIndices) );

trainingIdx = 1:idx;
trainingData = imagepaths(shuffledIndices(trainingIdx),:);
trainLabels = labels(shuffledIndices(trainingIdx),:);


validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationData = imagepaths(shuffledIndices(validationIdx),:);
validationLabels = labels(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testData = imagepaths(shuffledIndices(testIdx),:);
testLabels = labels(shuffledIndices(testIdx),:);

%% Saving off data
save('featExtractData.mat');

%% Load Data
load("featExtractData.mat")

%% Generate datastores
imgTrain = imageDatastore(trainingData, "Labels",trainLabels);

imgValidation = imageDatastore(validationData,"Labels",validationLabels);

imgTest = imageDatastore(testData,"Labels",testLabels);

YTrain = imgTrain.Labels;
YTest = imgTest.Labels;

%% Feature Extraction

net = resnet101;

inputSize = net.Layers(1).InputSize;

imageAugmenter = imageDataAugmenter('RandXTranslation', [-20 20], 'RandYTranslation', [-20 20], 'RandRotation', [-180 180], 'RandScale', [0.5 2]);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imgTrain, 'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imgValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imgTest);

layer = 'pool5';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
featuresValidate = activations(net,augimdsValidation,layer,'OutputAs','rows');

%% Checkpoint save
save('features.mat');

%% Checkpoint load
load('features.mat')


%% Train multi-class SVM
classifier = fitcecoc(featuresTrain,YTrain);

YPred = predict(classifier,featuresTest);
accuracy = mean(YPred == YTest)

