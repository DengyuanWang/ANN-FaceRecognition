function accuracy = main()
%     InputLayer = 10;
%     HiddenLayers = [100 100 100]';
%     OutputLayer = [62];
% X and Y need to be 1*n1 and 1*n2
    if(exist('Saver.mat','file')~=2)
        HR = HandWriting_recognition();
        save('Saver.mat','HR','-v7.3');
    else
        load('Saver.mat');
    end
    accuracy = HR.Five_fold_Cross_validation();

%     X = rand(1,2000);Y = zeros(1,62);
%     Y(1) =1;
%     InputLayer = size(X,2);
%     HiddenLayers = [100  100 100]';
%     OutputLayer =  size(Y,2);
%     LearningRate = 0.01;
%     net1 = ANN(InputLayer, HiddenLayers, OutputLayer,LearningRate);
%     
%     train_set = repmat({[{X} {Y}]},100,1);
%     batch_size = 10;
%     Batches = get_batches(train_set,batch_size);
%     
%     for i=1:length(Batches)
%         delta = repmat({0},batch_size,1);
%         batch = Batches{i};
%         parfor j=1:batch_size
%             tmp = batch{j}
%             t = net1.backpropagation(tmp{1},tmp{2});
%             delta{j} = t.get_deltatheta();
%         end
%         sample = delta{1};
%         result = repmat({0},size(sample));
%         for j=1:length(delta)
%             sample = delta{j};
%             for k=1:length(sample)
%                 result{k} = result{k}+ sample{k}./batch_size;
%             end
%         end
%         net1 = net1.set_theta(result);
%     end
%     Y = net1.predixt(X)
%     function Batches = get_batches(dataset,batchz_size)
%         len = ceil(size(dataset,1)/batchz_size);
%         lenmax = size(dataset,1);
%         Batches = repmat({0},len,1);
%         parfor i1=1:len
%             startindex = batchz_size*(i1-1)+1;
%             endindex = min(batchz_size*(i1),lenmax);
%             Batches{i1} = dataset(startindex:endindex,:);
%         end
%     end
end