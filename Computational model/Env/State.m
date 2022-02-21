classdef State
    properties
        state_index = -1;
        reward, is_terminal;
    end
    
    methods
        function obj = State(reward_, is_terminal_, state_index_)
            obj.reward = reward_;
            obj.is_terminal = is_terminal_;
            if nargin == 3
                obj.state_index = state_index_;        
            end
        end
    end
end