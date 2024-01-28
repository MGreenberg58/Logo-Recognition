clear;
clc;

ds = imageDatastore("C:\Users\brelanre\OneDrive - Rose-Hulman Institute of Technology\Documents\MATLAB\CSSE463\Transportation", ...
    FileExtensions=[".jpg"], ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");
nImages = numel(ds.Files);
images = ds.Files;

ds1 = imageDatastore("C:\Users\brelanre\OneDrive - Rose-Hulman Institute of Technology\Documents\MATLAB\CSSE463\Transportation", ...
    FileExtensions=[".xml"], ...
    IncludeSubfolders=true);
xmls = ds1.Files;

imagepaths = strings(nImages,1);
bb = {nImages,1};
for i = 1:nImages
    imagepaths(i) = string(images(i));
   
    name = xmls{i};
    xml = readstruct(name).object.bndbox;
    bb(i,:) = {[xml.xmin xml.ymin xml.xmax xml.ymax]};
end 

bb = bb(:,1)
%% Generate training, validation, and test sets
rng("default");
shuffledIndices = randperm(nImages);
idx = floor(0.6 * length(shuffledIndices) );

trainingIdx = 1:idx;
trainingData = imagepaths(shuffledIndices(trainingIdx),:);
trainingbb = table(bb(shuffledIndices(trainingIdx),:));

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationData = imagepaths(shuffledIndices(validationIdx),:);
validationbb = table(bb(shuffledIndices(validationIdx),:));

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testData = imagepaths(shuffledIndices(testIdx),:);
testbb = table(bb(shuffledIndices(testIdx),:));

%% Saving off data
save('RegionData.mat');

%% Load Data
load("RegionData.mat")

%% Generate datastores
imgTrain = imageDatastore(trainingData);
boxTrain = boxLabelDatastore(trainingbb);

imgValidation = imageDatastore(validationData);
boxValidation = boxLabelDatastore(validationbb);

imgTest = imageDatastore(testData);
boxTest = boxLabelDatastore(testbb);

trainingData = combine(imgTrain,boxTrain);
validationData = combine(imgValidation,boxValidation);
testData = combine(imgTest,boxTest);

%% Validate training set
validateInputData(trainingData);
validateInputData(validationData);
validateInputData(testData);

%% Display example image
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,"Rectangle",bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% Prep network and Augment Data
inputSize = [227 227 3];
className = "logo";
rng("default")
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 9;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);
area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");

anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    anchors(7:9,:)
    };
detector = yolov4ObjectDetector("csp-darknet53-coco",className,anchorBoxes,InputSize=inputSize);
augmentedTrainingData = transform(trainingData,@augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},"rectangle",data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,BorderSize=10)
options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.001,...
    LearnRateSchedule="none",...
    MiniBatchSize=100,...
    L2Regularization=0.0005,...
    MaxEpochs=25,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=20,...
    ValidationFrequency=1000,...
    CheckpointPath=tempdir,...
    ValidationData=validationData);
[detector,info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);

%% Train the Network















