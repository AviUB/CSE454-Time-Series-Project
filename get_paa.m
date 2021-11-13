% get piecewise aggregate approximation from a dataset
% input: dataset, number of samples, number of observations, number of segments
% output: mean sample
function paa = get_paa(d, ns, dt, c)
    % get segment length
    slen=dt/c;
    slen=ceil(slen);

    % go through each sample
    for i=1:ns
        % for each window
        for N=1:c
            % empty vector to collect everything
            sam=[];
            samCnt=1;
            % if within range of observation (big window)
            if (N-1)*slen<dt
                % set current window's bounds 
                bl=(N-1)*slen;
                bu=N*slen;
                % see if observation is inside the window
                for j=1:dt
                    if j>bl
                        if j<= bu
                            % valid observation, store it, increment count
                            sam(samCnt)=d(i,j);
                            samCnt=samCnt+1;
                        end
                    end

                end
            end
            % after all observations are taken for each segment,
            % take the average and store it in the paa
            paa(i,N)=mean(sam);
        end
    end
end