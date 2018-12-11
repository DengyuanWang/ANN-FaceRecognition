function HR = main()
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
    HR.validation()
    save('Saver.mat','HR','-v7.3');
end