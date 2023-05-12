classdef eqs < handle
    properties
        x_start = [0,0];
        x_goal = [0,0];
        state_dims = 4;
        input_dims = 2;
        A = [[0,0,1,0];
             [0,0,0,1];
             [0,0,0,0];
             [0,0,0,0];];
        B = [[0, 0];
             [0, 0];
             [1, 0];
             [0, 1];];
        R = [[1, 0];
             [0, 1];];
        c = [[0]; [0]; [0]; [0];];
        tau = sym("tau");
        t = sym("t", "positive");
        ts = sym("ts", 'real');
        x = sym("x",'positive');
        x0 = [];
        x1 = [];
        G = 0;
        G_f = 0;
        x_bar = 0;
        d = 0;
        sol = 0;
        tau_f = 0;
        cost = 0;
        time = 0;
        states = 0;
        inputs = 0;
        state_free = 0;
        input_free = 0;
%        max_radius = 100;
        state_limits = ...
        [0,100;
        0,100;
        -10,10;
        -10,10];
        input_limits = ...
        [ -5, 5;
        -5, 5];
        d_ret = 0;
        x_rand_children = 0;
        freespace = 0;
        gamma_radius = 100;
        radius = sym("r","positive");
        tau_r = 0;
        vr = 0;
        r = 0;


    end
    methods     
        function obj = eqs(st_dims, in_dims, state_limits, input_limits, obs, x_start, x_goal)
            obj.x_start = x_start;
            obj.x_goal = x_goal;
            obj.state_dims = st_dims;
            obj.input_dims = in_dims;
            obj.x0 = sym('x0',[st_dims,1]);
            assume(obj.x0, "real");
            obj.x1 = sym('x1',[st_dims,1]);
            assume(obj.x1, "real");
            [obj.G,obj.x_bar,obj.d,obj.sol] = aux(obj);
            obj.G_f = matlabFunction(obj.G);
            states_ = obj.sol(1:obj.state_dims);
            obj.states = matlabFunction(states_);

            inputs_ = obj.R\obj.B'*obj.sol(obj.state_dims+1:2*obj.state_dims,:);
            obj.inputs = matlabFunction(inputs_);

            cost_ = int(1+inputs_'*obj.R*inputs_, obj.t, 0, obj.ts);
            obj.cost = matlabFunction(cost_);

            p1 = [];
            it = 0;
            tau_out_ = 1-2*(obj.A*obj.x1+obj.c)'*obj.d-obj.d'*obj.B/obj.R*obj.B'*obj.d;
            while it < 20 && length(p1) <= 1 %just so mupad knows it's a polynomial
                p1 = feval(symengine,'coeff',simplifyFraction(tau_out_*obj.t^it), obj.t, 'All');
                it = it+1;
            end
            p([obj.x0',obj.x1']) = fliplr(p1);
            obj.tau_f = matlabFunction(p);
            radius = 1;
            obstacles = obs;
            obs_area = 0;
            
            lobs = length(obstacles(1));
            if lobs ~= 4
                obs_area = obs_area + obstacles(3)*obstacles(4);
        
            else
                for jj=1:length(obstacles)
                    obs_area = obs_area + obstacles(jj,3)*obstacles(jj,4);
                end
            end
            obj.freespace = (obj.state_limits(1,2)*obj.state_limits(2,2) - obs_area)*400; %CAMBIARE IN BASE A ENVIRONMENT
            obj.gamma_radius = (2^obj.state_dims)*(1 + 1/obj.state_dims)*obj.freespace;
            
            p2= [];
            it2 = 0;
            diffdetG = diff(det(obj.G*(obj.radius - obj.t)),obj.t);
            while it2 < 20 && length(p2) <= 1 %just so mupad knows it's a polynomial
                p2 = feval(symengine,'coeff',simplifyFraction(diffdetG*obj.t^it2), obj.t, 'All');
                it2 = it2+1;
            end
            obj.G_f = fliplr(p2);
            diffdetGroots = roots(obj.G_f);
            for jj=1:length(diffdetGroots)
                if diffdetGroots(jj) ~= 0 && diffdetGroots(jj) ~= obj.radius
                    obj.tau_r = diffdetGroots(jj);
                end
            end
            obj.vr = 4.935*det(subs(obj.G,obj.t,obj.tau_r)*(obj.radius-obj.tau_r));
            obj.state_free = @(state, time_range)(ball_is_state_free(state, obj.state_limits, obstacles, radius, time_range));
            obj.input_free = @(input, time_range)(is_input_free(input, obj.input_limits, time_range));
        end
        function  [G, x_bar, d, sol] = aux(obj)
            G = int(expm(obj.A*(obj.t-obj.x))*obj.B/obj.R*obj.B'*expm(obj.A'*(obj.t-obj.x)), obj.x, 0, obj.t);
            x_bar = expm(obj.A*obj.t)*obj.x0+int(expm(obj.A*(obj.t-obj.x))*obj.c, obj.x, 0, obj.t);
            d = G\(obj.x1-x_bar);
            % disp(d)
            sol = expm([obj.A, obj.B/obj.R*obj.B';zeros(obj.state_dims), -obj.A']*(obj.t-obj.ts))*[obj.x1;subs(d,obj.t,obj.ts)]+ ...
                int(expm([obj.A, obj.B/obj.R*obj.B';zeros(obj.state_dims), -obj.A']*(obj.t-obj.x))*[obj.c; zeros(obj.state_dims,1)],obj.x,obj.ts,obj.t);

        end
        function tau_out = taustar(obj, x0_, x1_)
            % x0_
            % x1_
            in = num2cell([x0_', x1_']);
            tau_out = obj.tau_f(in{:});
            tau_out = roots(tau_out);
            tau_out = tau_out(imag(tau_out)==0);
            tau_out = min(tau_out(tau_out>=0));
            
        end
        function [cost, time] = eval_cost(obj, n1, n2, time)
            % if ~exist('time','var') || isempty(time)
            if isempty(time)
                time = obj.taustar(n1, n2);
            end

            in = num2cell([time, n1', n2']);
            cost = obj.cost(in{:});
            
        end
        function [states_, inputs_] = evaluate_states_and_inputs(obj, node, x_rand, time)
            if isempty(time)
                time = obj.taustar(x0_, x1_);
            end
            in = num2cell([time, node', x_rand']);
            states_ = @(t)obj.states(t, in{:});
            inputs_ = @(t)obj.inputs(t, in{:});
        end
        function ret = ChooseParent(obj, V, costs, times, x_rand, goal_cost)
            
            %goal_cost è c_best!
            min_idx = 1;
            min_cost = inf;
            min_time = inf;
            idx = 1;
            sV = size(V);
            
            sVlog = sV(1)+1;
            p3= [];
            it3 = 0;
            r_pol = obj.vr - (obj.gamma_radius*log(sVlog)/sVlog)^2;
            obj.r = double(solve(r_pol,obj.radius));
            while idx <= sV(1)
                node = V(idx,:)';
                
                % if is_terminal(node_idx)
                %     continue;
                [cost, time] = obj.eval_cost(node,x_rand, []);
                if cost < obj.r && costs(idx)+cost < min_cost && costs(idx)+cost < goal_cost
                    [states, u] = evaluate_states_and_inputs(obj,node,x_rand,time);
                    if  obj.state_free(states,[0,time]) && obj.input_free(u,[0,time])
                        min_idx = idx;
                        min_cost = costs(idx)+cost;
                        min_time = times(idx)+time;
                        % continue;
                    end
                end
                %if we arrive here it means there is no trajectory from node to new x_i
                %however child nodes might be able to form a connection
                idx = idx + 1;
            end
            ret = [min_cost, min_time, min_idx];
        end
        function d_ret = Rewire(obj, V, costs, times, costs2goal, times2goal, nearsgoal, nodeschildren, x_rand, x_rand_cost, x_rand_time, goal_cost, c_best, t_best, x_best)

            V = V';
            x_rand_children = {};
            ones_vec = ones(1,4);

            stack(1).index = 1;
            stack(1).improvement = 0;
            stack(1).time_improvement = 0;
            stack(1).parent = NaN;

            while ~isempty(stack)

                node = stack(end);
                stack = stack(1:end-1);

                x_curr = V(:, node.index);


                costs(node.index) = costs(node.index)-node.improvement;
                times(node.index) = times(node.index)-node.time_improvement;
                if nearsgoal(node.index)
                    if costs(node.index) + costs2goal(node.index)< c_best
                        c_best = costs(node.index) + costs2goal(node.index);
                        t_best = times(node.index) + times2goal(node.index);
                        x_best = node.index;
                    end
                    continue;
                end
                diff = node.improvement;
                time_diff = node.time_improvement;
              
                [partial_cost, partial_time] = obj.eval_cost(x_rand, x_curr, []);
         
                if partial_cost < obj.r
                    new_cost = partial_cost + x_rand_cost;
                    if new_cost < costs(node.index)
                        [states, u] = obj.evaluate_states_and_inputs(x_rand,x_curr,partial_time);
                        if obj.state_free(states,[0,partial_time]) && obj.input_free(u,[0,partial_time])
    
                            new_time = partial_time + x_rand_time;
                            diff = diff + costs(node.index) - new_cost; 
                            time_diff = time_diff + times(node.index) - new_time;
                            costs(node.index) = new_cost;
                            times(node.index) = new_time;
                            
                            x_rand_children = [x_rand_children, [node.index, new_cost, new_time, node.parent]];
                        end
                    end
                end
               
                for jj=1:length(nodeschildren{node.index})
                   lm = V == cell2mat(nodeschildren{node.index}(jj));
                   r = ones_vec*lm == 4;
                   index = find(r);
                   
                   if isempty(index)
                       continue
                   end

                   stack(end+1).index = index;
                   stack(end).parent = node.index;
                   stack(end).improvement = diff;
                   stack(end).time_improvement = time_diff;
                   
                end
      
              
            end
            ret_rewire = {c_best,t_best,x_best,x_rand_children};
            obj.x_rand_children = x_rand_children;
            names = ["c_best" "t_best" "x_best" "x_rand_children"];
            d_ret = struct("c_best",c_best,"t_best",t_best,"x_best",x_best,"costs",costs,"times",times);
            
            % d_ret = dictionary(names,ret_rewire);
            % obj.d_ret = d_ret
            
        end
        

    end
end
