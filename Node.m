classdef Node
    properties
        Layer_index;
        LearningRate;
        Weight;%n*1 matrix
        In;
        Out;
        Stashed_Error;
        Stashed_Weight;
    end
    methods
        function obj = Node(outdimension,Layer_index,out_tag)
            obj.Layer_index = Layer_index;
            if nargin>2%more than 2 input arguments
                obj.Weight = ones(outdimension,1);
                obj.Stashed_Weight = obj.Weight;
            else
                obj.Weight = rand(outdimension,1);
                obj.Stashed_Weight = obj.Weight;
            end
            
            obj.In = 0;
            obj.Out = activation(obj.In);
            obj.Stashed_Error = 0;
        end
        function obj = update(obj,in)
            % update weight
            obj.Weight = obj.Stashed_Weight;
            obj.In = in;
            obj.Out = activation(obj.In);
        end
        function obj = subWeight(obj,delta_weights)
             obj.Stashed_Weight = obj.Stashed_Weight - delta_weights{1};
        end
    end
end
function out = activation(in)%sigmoid
    out = 1/(1+exp(-in));
end