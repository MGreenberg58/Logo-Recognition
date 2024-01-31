clear;
clc;

%jadon's filepath note: I am using the images after running that python
%script, and I would imagine that your accuracy will drop/be lower than mine 
% if you haven't run that script yet due to multiple labels for the same class

filePath = "TransportationCleaned/";

ds = imageDatastore(filePath, ...
    FileExtensions=[".jpg"], ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");
nImages = numel(ds.Files);
images = ds.Files;
labels = ds.Labels;


%% Generate training, validation, and test sets
rng("default");

% Get a list of subfolders in the main folder
subfolders = dir(filePath);
subfolders = subfolders([subfolders.isdir]); % Keep only directories
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % Remove '.' and '..'

trainingPercentage = 60;
validationPercentage = 10;
testingPercentage = 30;

trainingData = {};
validationData = {};
testingData = {};

trainingLabels = {};
validationLabels = {};
testingLabels = {};

for i = 1:length(subfolders)
    subfolder = fullfile(filePath, subfolders(i).name);
    
    % Get a list of files in the current subfolder
    imgFiles = dir(fullfile(subfolder, '*.jpg'));
    
    % Randomly shuffle the list of files
    perm = randperm(length(imgFiles));

    % Calculate the number of files for each set
    numFiles = length(imgFiles);
    numTraining = round(trainingPercentage/100 * numFiles);
    numValidation = round(validationPercentage/100 * numFiles);
    numTesting = numFiles - numTraining - numValidation;
    
    % Divide the files into training, validation, and testing sets
    
    for j = 1:numTraining
        trainingData = [trainingData; fullfile(subfolder, imgFiles(perm(j)).name)'];
        trainingLabels = [trainingLabels; labels(perm(j))];
    end

    for j = numTraining+1:numTraining+numValidation
        validationData = [validationData; fullfile(subfolder, imgFiles(perm(j)).name)'];
        validationLabels = [validationLabels; labels(perm(j))];
    end

    for j = numTraining+numValidation+1:numFiles
        testingData = [testingData; fullfile(subfolder, imgFiles(perm(j)).name)'];
        testingLabels = [testingLabels; labels(perm(j))];
    end

    disp(['Processing subfolder: ' subfolders(i).name]);
    disp(['  Training set: ' num2str(numTraining) ' files']);
    disp(['  Validation set: ' num2str(numValidation) ' files']);
    disp(['  Testing set: ' num2str(numTesting) ' files']);

end

%% Generate datastores
imgTrain = imageDatastore(trainingData);

imgValidation = imageDatastore(validationData);

imgTest = imageDatastore(testingData);

%% Saving off data
save('featExtractData.mat');

%% Load Data
load("featExtractData.mat")

%% Load net and augment inputs 

net = resnet101;

inputSize = net.Layers(1).InputSize;

%% Augment Inputs

%you could experiment here with different augmentations maybe?
imageAugmenter = imageDataAugmenter('RandXTranslation', [-20 20], 'RandYTranslation', [-20 20], 'RandRotation', [-180 180], 'RandScale', [0.5 2]);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imgTrain, 'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imgValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imgTest);

%% Feature Extraction with global pooling layer: pool5 features

layer = 'pool5';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
featuresValidate = activations(net,augimdsValidation,layer,'OutputAs','rows');

%% Checkpoint save
save('pool5features.mat');

%% Feature Extraction with early activation layer 1: res3b3_relu features

layer = 'res3b3_relu';
featuresTrain = squeeze(mean(activations(net,augimdsTrain,layer),[1 2]))';
featuresTest = squeeze(mean(activations(net,augimdsTest,layer),[1 2]))';
featuresValidate = squeeze(mean(activations(net,augimdsValidation,layer),[1 2]))';

%% Checkpoint save
save('res3b3features.mat');

%% Feature Extraction with early activation layer 2: res4b6_relu features

layer = 'res4b6_relu';
featuresTrain = squeeze(mean(activations(net,augimdsTrain,layer),[1 2]))';
featuresTest = squeeze(mean(activations(net,augimdsTest,layer),[1 2]))';
featuresValidate = squeeze(mean(activations(net,augimdsValidation,layer),[1 2]))';

%% Checkpoint save
save('res4b6features.mat');

%% Feature Extraction with early activation layer 3: res4b12_relu features

layer = 'res4b12_relu';
featuresTrain = squeeze(mean(activations(net,augimdsTrain,layer),[1 2]))';
featuresTest = squeeze(mean(activations(net,augimdsTest,layer),[1 2]))';
featuresValidate = squeeze(mean(activations(net,augimdsValidation,layer),[1 2]))';

%% Checkpoint save
save('res4b12features.mat');

%% Feature Extraction with early activation layer 4: res4b18_relu features

layer = 'res4b18_relu';
featuresTrain = squeeze(mean(activations(net,augimdsTrain,layer),[1 2]))';
featuresTest = squeeze(mean(activations(net,augimdsTest,layer),[1 2]))';
featuresValidate = squeeze(mean(activations(net,augimdsValidation,layer),[1 2]))';

%% Checkpoint save
save('res4b18features.mat');

%% Checkpoint load
%load('pool5features.mat')


%% Train multi-class SVM
classifier = fitcecoc(featuresTrain,trainingLabels);

YPred = predict(classifier,featuresTest);
accuracy = mean(YPred == testingLabels)

