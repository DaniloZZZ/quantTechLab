classdef f_func
    properties
        T = 0.6;
        t=0;
        r=0;
    end
   
    methods
        function obj = f_func()
            T = 0.6
            obj.t = sqrt(T)
            obj.r = sqrt(1-T)
        end

        function y = a01f(obj,al,thet)
            y = exp(1i*thet)*...
            [
                obj.t obj.r*exp(1i*al) 0;
                -obj.r*exp(1i*al) obj.t 0;
                0 0 exp(-1i*thet)
            ]
        end
    end
end
