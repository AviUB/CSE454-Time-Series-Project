% plot a piecewise aggregate approximation from a dataset
% input: dataset, number of samples, number of observations, 
%        number of segments, index of sample for paa is to be plotted
% output: mean sample
function  paaplot(paa, data, ns, dt, c, i)
    slen=dt/c;      % segment length
    slen=ceil(slen);
    paax=[];        % paa observation vectors
    paay=[];
    z=1;            % index of observation
    n=1;            % initial segment
    
    paax(z)=0;      % initial conditions
    paay(z)=paa(i,n);
    z=z+1;          % increment to next index 
    % update observations
    % need to know 2 (x,y) at every segment for piecewise function shape
    for n=2:c
        paax(z)=paax(z-1)+slen;
        paay(z)=paa(i,n-1);
        z=z+1;
        paax(z)=paax(z-1);
        paay(z)=paa(i,n);
        z=z+1;
    end
    % last segment
    paax(z)=paax(z-1);
    paay(z)=paa(i,n);
    z=z+1;
    paax(z)=paax(z-1)+slen;
    paay(z)=paa(i,n);
    % plot each observation's (x,y)
    plot(paax,paay);
    hold on
    t=linspace(0,dt,dt);
    scatter(t,data(i,:),"filled")
end